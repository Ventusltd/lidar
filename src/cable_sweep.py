# SPDX-License-Identifier: Apache-2.0
"""Cable sweep: how long is a buried cable over real ground, counted two ways?

Two channels a case, photons back (the electron/positron pattern of pair_gpu.py and
earthworks_pair.py). Where the channels disagree by more than 0.1 % a photon is COUNTED,
never suppressed, and the worst routes are kept with their places.

A route is a plan polyline (straight, or 1-4 bends) whose corners are filleted with circular
arcs, as web/world/cable-route.mjs does: every corner asks for the minimum bend radius R; a
corner whose legs are too short gets the largest radius its legs allow (a short leg is shared
between its two corners in proportion to what each asks) and is counted as a violation. The
cable lies a fixed depth below the ground directly under the centreline (0.9 m at 33 kV,
1.05 m at 132 kV), so its slope is the ground's slope along the route.

R comes from cables.json (graphics-engines-open-source-world/web/world/data): the cable of the
voltage and conductor size (1000 mm2), a verified OD preferred, times the governing (largest)
installation multiple listed for that voltage.

  electron   dense sampling: the filleted centreline sampled every <= 0.05 m of chainage, each
             sample dropped onto the bilinear ground, the 3D chords summed.
  positron   analytic plan length (straights plus r |theta|) plus a vertical correction
             integral of (sqrt(1 + g^2) - 1) ds, g = dz/ds. Every piece is cut exactly where it
             crosses a grid line; inside one cell a straight's slope is linear in s, so its
             integral is closed form; an arc's is 6-point Gauss-Legendre on a smooth integrand.

Ground is the bilinear surface of the float64 source (source.npy, SW origin in source.json).
Seeded (default_rng), bounded (--routes or --seconds, whichever first). Witness: the first
routes recomputed in NumPy on the CPU, agreeing with the card within 1e-9.

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/cable_sweep.py --tiles DIR
      [--routes 40000] [--seconds 60] [--seed N] [--cpu] [--cables PATH]
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card, no CuPy: the same arithmetic runs in NumPy and says so
    cp = None

SEED = 20260926
TOL = 1e-3            # relative disagreement that makes a photon (0.1 %)
DS_E = 0.05           # electron sampling step along chainage (m)
WIT_TOL = 1e-9
KEEP = 8
NB = 4                # most bends a route has
NP = 2 * NB + 1       # pieces a route: line, arc, line, ... line
RECEIPT = "cable_sweep_receipt.json"
CSA = 1000
DEPTH = {33: 0.9, 132: 1.05}
CABLES = os.path.join(os.path.expanduser("~"), "Documents", "GitHub", "graphics-engines-open-source-world",
                      "web", "world", "data", "cables.json")
GL_X, GL_W = np.polynomial.legendre.leggauss(6)


# ---------------------------------------------------------------- ground (as earthworks_pair.py)
def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


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
    """source.npy (float64) with its SW origin and spacing from source.json beside it."""
    grid = np.load(os.path.join(tiles_dir, "source.npy")).astype(np.float64)
    meta = json.load(open(os.path.join(tiles_dir, "source.json"), encoding="utf-8"))
    pick = lambda *ks: next((meta[k] for k in ks if k in meta), None)
    oe, on = pick("origin_e_m", "origin_e", "sw_e", "x0"), pick("origin_n_m", "origin_n", "sw_n", "y0")
    sp = pick("spacing_m", "spacing", "cell_m", "res_m")
    if oe is None or on is None or sp is None:
        raise ValueError(f"{tiles_dir}/source.json: need SW origin and spacing_m")
    if str(meta.get("rows", "south_to_north")).startswith("north"):
        grid = grid[::-1]
    return Ground(grid, float(oe), float(on), float(sp), xp)


# ---------------------------------------------------------------- bend rule
def min_radius(cables, kv, csa=CSA, when="installation"):
    """Governing minimum bend radius (m) for a single-core cable of kv and csa, with its sources."""
    cands = [c for c in cables["cables"] if c["voltage_kv"] == kv and c["csa_mm2"] == csa]
    if not cands:
        raise ValueError(f"cables.json: no {kv} kV {csa} mm2 cable")
    cab = sorted(cands, key=lambda c: (c.get("od_status") != "verified", "spec" in c["id"]))[0]
    rules = [r for r in cables["rules"] if r["voltage_kv"] == kv and r["when"] == when and r.get("multiple_of_od")]
    if not rules:
        raise ValueError(f"cables.json: no {when} rule at {kv} kV")
    rule = max(rules, key=lambda r: r["multiple_of_od"])
    return dict(voltage_kv=kv, cable_id=cab["id"], od_mm=cab["od_mm"], od_status=cab.get("od_status"),
                multiple_of_od=rule["multiple_of_od"], construction=rule["construction"], when=when,
                rule_status=rule.get("status"), rule_source=rule["source"],
                radius_m=rule["multiple_of_od"] * cab["od_mm"] / 1000.0, depth_m=DEPTH.get(kv))


# ---------------------------------------------------------------- plan geometry (host, NumPy)
def plan(x0, y0, h0, legs, turns, R):
    """Filleted plan geometry of n routes.

    x0, y0, h0 (n,), legs (n, NB+1) leg lengths (0 past the last), turns (n, NB) signed radians
    (0 past the last bend), R (n,) requested radius. Returns pieces (n, NP) with start x, y,
    heading, curvature and length, plus fitted radii, violations and the analytic plan length.
    """
    n = len(x0)
    tturn = np.tan(np.abs(turns) / 2)
    want = np.zeros((n, NB + 2)); want[:, 1:NB + 1] = R[:, None] * tturn
    cap = want.copy()
    for k in range(NB + 1):
        t0, t1, L = want[:, k], want[:, k + 1], legs[:, k]
        over = t0 + t1 > L
        f0 = np.where(t0 + t1 > 0, t0 / np.where(t0 + t1 > 0, t0 + t1, 1), 0.5)
        cap[:, k] = np.where(over, np.minimum(cap[:, k], L * f0), cap[:, k])
        cap[:, k + 1] = np.where(over, np.minimum(cap[:, k + 1], L * (1 - f0)), cap[:, k + 1])
    t = cap[:, 1:NB + 1]
    bend = turns != 0
    r = np.where(bend, t / np.where(bend, tturn, 1), 0.0)
    viol = bend & (r < R[:, None] * (1 - 1e-9))
    P = {k: np.zeros((n, NP)) for k in ("x", "y", "h", "kap", "L")}
    x, y, h = x0.astype(float).copy(), y0.astype(float).copy(), h0.astype(float).copy()
    tpad = np.concatenate([np.zeros((n, 1)), t, np.zeros((n, 1))], 1)
    for k in range(NB + 1):
        Ll = np.maximum(legs[:, k] - tpad[:, k] - tpad[:, k + 1], 0.0)
        P["x"][:, 2 * k], P["y"][:, 2 * k], P["h"][:, 2 * k], P["L"][:, 2 * k] = x, y, h, Ll
        x, y = x + Ll * np.cos(h), y + Ll * np.sin(h)
        if k < NB:
            th, rk = turns[:, k], r[:, k]
            La = rk * np.abs(th)
            kap = np.where(rk > 0, np.sign(th) / np.where(rk > 0, rk, 1), 0.0)
            P["x"][:, 2 * k + 1], P["y"][:, 2 * k + 1], P["h"][:, 2 * k + 1] = x, y, h
            P["kap"][:, 2 * k + 1], P["L"][:, 2 * k + 1] = kap, La
            x, y = arc_point(x, y, h, kap, La, np)
            h = h + th
    P["S0"] = np.concatenate([np.zeros((n, 1)), np.cumsum(P["L"], 1)[:, :-1]], 1)
    P["L2d"] = P["L"].sum(1)
    return P, r, viol


def arc_point(x, y, h, kap, u, xp):
    """Point at chainage u along a piece starting at (x, y) heading h with curvature kap."""
    half = kap * u / 2
    ch = u * xp.sinc(half / np.pi)          # chord length; u on a straight
    return x + ch * xp.cos(h + half), y + ch * xp.sin(h + half)


def ragged(xp, counts):
    """Owner and local index of a ragged range: counts (m,) -> (owner, j) of length counts.sum()."""
    ends = xp.cumsum(counts)
    tot = int(ends[-1]) if len(counts) else 0
    k = xp.arange(tot)
    owner = xp.searchsorted(ends, k, side="right")
    return owner, k - (ends - counts)[owner]


# ---------------------------------------------------------------- electron
def electron(P, depth, G, ds=DS_E):
    """3D length from dense samples, and the 2D chord sum (a plan check), per route."""
    xp = G.xp
    n = len(P["L2d"])
    L2 = xp.asarray(P["L2d"])
    ns = xp.ceil(L2 / ds).astype(xp.int64) + 1
    own, j = ragged(xp, ns)
    s = j * (L2 / (ns - 1))[own]
    S0 = xp.asarray(P["S0"])[own]
    pid = (s[:, None] >= S0[:, 1:]).sum(1)
    g = lambda k: xp.asarray(P[k])[own, pid]
    u = s - S0[xp.arange(own.size), pid]
    x, y = arc_point(g("x"), g("y"), g("h"), g("kap"), u, xp)
    z = G.h(x, y) - xp.asarray(depth)[own]
    same = own[1:] == own[:-1]
    dx, dy, dz = x[1:] - x[:-1], y[1:] - y[:-1], z[1:] - z[:-1]
    d2 = xp.where(same, xp.sqrt(dx * dx + dy * dy), 0)
    d3 = xp.where(same, xp.sqrt(dx * dx + dy * dy + dz * dz), 0)
    L3 = to_host(xp.bincount(own[:-1], weights=d3, minlength=n))
    C2 = to_host(xp.bincount(own[:-1], weights=d2, minlength=n))
    return L3, C2, int(own.size)


# ---------------------------------------------------------------- positron
def _F(u, xp):
    return (u * xp.sqrt(1 + u * u) + xp.arcsinh(u)) / 2


def crossings(P, G):
    """Breakpoints (piece id, s) where each piece meets a grid line, plus its two ends."""
    xp, sp = G.xp, G.sp
    flat = {k: xp.asarray(P[k]).ravel() for k in ("x", "y", "h", "kap", "L")}
    live = xp.nonzero(flat["L"] > 0)[0]
    x0, y0, h0, kap, L = (flat[k][live] for k in ("x", "y", "h", "kap", "L"))
    arc = kap != 0
    PID, S = [live, live], [xp.zeros_like(L), L]
    # straights: s = (k sp - x0) / ux between the ends
    ln = xp.nonzero(~arc)[0]
    x1, y1 = arc_point(x0[ln], y0[ln], h0[ln], kap[ln], L[ln], xp)
    for a0, a1, c in ((x0[ln], x1, xp.cos(h0[ln])), (y0[ln], y1, xp.sin(h0[ln]))):
        lo, hi = xp.floor(xp.minimum(a0, a1) / sp), xp.floor(xp.maximum(a0, a1) / sp)
        o, j = ragged(xp, (hi - lo).astype(xp.int64))
        s = ((lo[o] + 1 + j) * sp - a0[o]) / c[o]
        PID.append(live[ln][o]); S.append(s)
    # arcs: polar angle alpha = alpha0 + kap s about the centre
    ac = xp.nonzero(arc)[0]
    ka, xa, ya, ha, La = kap[ac], x0[ac], y0[ac], h0[ac], L[ac]
    r = 1 / xp.abs(ka)
    cx, cy = xa - xp.sin(ha) / ka, ya + xp.cos(ha) / ka
    al0 = xp.arctan2(ya - cy, xa - cx)
    sg = xp.sign(ka)
    for c0, is_x in ((cx, True), (cy, False)):
        lo, hi = xp.floor((c0 - r) / sp), xp.floor((c0 + r) / sp)
        o, j = ragged(xp, (hi - lo).astype(xp.int64))
        v = xp.clip(((lo[o] + 1 + j) * sp - c0[o]) / r[o], -1, 1)
        base = xp.arccos(v) if is_x else xp.arcsin(v)
        for alpha in ((base, -base) if is_x else (base, np.pi - base)):
            s = xp.mod((alpha - al0[o]) * sg[o], 2 * np.pi) * r[o]
            PID.append(live[ac][o]); S.append(s)
    pid, s = xp.concatenate(PID), xp.concatenate(S)
    keep = (s >= 0) & (s <= flat["L"][pid])
    pid, s = pid[keep], s[keep]
    o1 = xp.argsort(s, kind="stable") if xp is np else xp.argsort(s)
    pid, s = pid[o1], s[o1]
    o2 = xp.argsort(pid, kind="stable") if xp is np else _stable_argsort_int(pid, xp)
    return pid[o2], s[o2], flat


def _stable_argsort_int(a, xp):
    """Stable argsort of a non-negative int array on CuPy: sort (a, position) packed in one int64."""
    pos = xp.arange(a.size, dtype=xp.int64)
    return xp.argsort(a.astype(xp.int64) * (1 << 34) + pos) if a.size < (1 << 34) else xp.argsort(a)


def positron(P, G):
    """Analytic plan length plus an exact per-cell vertical correction, per route."""
    xp, sp = G.xp, G.sp
    n = len(P["L2d"])
    pid, s, f = crossings(P, G)
    same = pid[1:] == pid[:-1]
    a, b, p = s[:-1][same], s[1:][same], pid[:-1][same]
    dl = b - a
    x0, y0, h0, kap = f["x"][p], f["y"][p], f["h"][p], f["kap"][p]
    # the cell of each interval, from its middle
    xm, ym = arc_point(x0, y0, h0, kap, (a + b) / 2, xp)
    j = xp.clip(xp.floor(xm / sp), 0, G.nx - 2).astype(xp.int64)
    i = xp.clip(xp.floor(ym / sp), 0, G.ny - 2).astype(xp.int64)
    g = G.g
    A = g[i, j]; B = g[i, j + 1] - A; C = g[i + 1, j] - A; D = g[i + 1, j + 1] - g[i + 1, j] - g[i, j + 1] + A
    # straights: dz/ds = p0 + q s inside one cell, integral closed form
    ln = kap == 0
    xs, ys = arc_point(x0, y0, h0, kap, a, xp)
    fx, fy = xs / sp - j, ys / sp - i
    ux, uy = xp.cos(h0), xp.sin(h0)
    p0 = (B * ux + C * uy + D * (ux * fy + uy * fx)) / sp
    q = 2 * D * ux * uy / (sp * sp)
    qs = xp.where(xp.abs(q * dl) > 1e-7, q, 1.0)
    exact = (_F(p0 + q * dl, xp) - _F(p0, xp)) / qs - dl
    gm = p0 + q * dl / 2
    small = gm * gm / (1 + xp.sqrt(1 + gm * gm)) * dl
    corr = xp.where(ln, xp.where(xp.abs(q * dl) > 1e-7, exact, small), 0.0)
    # arcs: 6-point Gauss-Legendre with the cell's gradient
    ai = xp.nonzero(~ln)[0]
    if ai.size:
        gx, gw = xp.asarray(GL_X), xp.asarray(GL_W)
        aa, bb = a[ai], b[ai]
        sq = (aa + bb)[:, None] / 2 + (bb - aa)[:, None] / 2 * gx[None]
        X, Y = arc_point(x0[ai, None], y0[ai, None], h0[ai, None], kap[ai, None], sq, xp)
        FX, FY = X / sp - j[ai, None], Y / sp - i[ai, None]
        hx = (B[ai, None] + D[ai, None] * FY) / sp
        hy = (C[ai, None] + D[ai, None] * FX) / sp
        ph = h0[ai, None] + kap[ai, None] * sq
        gg = hx * xp.cos(ph) + hy * xp.sin(ph)
        val = gg * gg / (1 + xp.sqrt(1 + gg * gg))
        corr[ai] = (val * gw[None]).sum(1) * (bb - aa) / 2
    V = to_host(xp.bincount(p // NP, weights=corr, minlength=n))
    return P["L2d"] + V, V, int(dl.size)


# ---------------------------------------------------------------- sweep
def draw(rng, n, G, rule, margin=3.0, leg=(3.0, 300.0)):
    """n seeded routes wholly inside the site: 0-4 bends (uniform), 33 or 132 kV (half each)."""
    W, H = G.sp * (G.nx - 1), G.sp * (G.ny - 1)
    out = {k: [] for k in ("x0", "y0", "h0", "legs", "turns", "kv")}
    have = 0
    while have < n:
        m = 2 * (n - have) + 16
        nb = rng.integers(0, NB + 1, m)
        kv = np.where(rng.random(m) < 0.5, 33, 132)
        legs = np.exp(rng.uniform(math.log(leg[0]), math.log(leg[1]), (m, NB + 1)))
        legs[np.arange(NB + 1)[None] > nb[:, None]] = 0
        turns = rng.uniform(np.radians(5), np.radians(150), (m, NB)) * rng.choice([-1, 1], (m, NB))
        turns[np.arange(NB)[None] >= nb[:, None]] = 0
        x0, y0, h0 = rng.uniform(margin, W - margin, m), rng.uniform(margin, H - margin, m), rng.uniform(0, 2 * np.pi, m)
        hv = h0[:, None] + np.concatenate([np.zeros((m, 1)), np.cumsum(turns, 1)], 1)
        vx = x0[:, None] + np.concatenate([np.zeros((m, 1)), np.cumsum(legs * np.cos(hv), 1)], 1)
        vy = y0[:, None] + np.concatenate([np.zeros((m, 1)), np.cumsum(legs * np.sin(hv), 1)], 1)
        ok = (vx.min(1) > margin) & (vy.min(1) > margin) & (vx.max(1) < W - margin) & (vy.max(1) < H - margin)
        ok &= np.cumsum(ok) <= n - have
        for k, v in (("x0", x0), ("y0", y0), ("h0", h0), ("legs", legs), ("turns", turns), ("kv", kv)):
            out[k].append(v[ok])
        have += int(ok.sum())
    D = {k: np.concatenate(v) for k, v in out.items()}
    D["R"] = np.where(D["kv"] == 33, rule[33]["radius_m"], rule[132]["radius_m"])
    D["depth"] = np.where(D["kv"] == 33, rule[33]["depth_m"], rule[132]["depth_m"])
    return D


def pair(D, G):
    P, r, viol = plan(D["x0"], D["y0"], D["h0"], D["legs"], D["turns"], D["R"])
    e3, c2, ns = electron(P, D["depth"], G)
    p3, corr, ni = positron(P, G)
    rel = np.abs(e3 - p3) / np.maximum(np.maximum(e3, p3), 1e-12)
    return dict(P=P, r=r, viol=viol, e3=e3, c2=c2, p3=p3, corr=corr, rel=rel, samples=ns, intervals=ni)


def load_cables(path):
    return json.load(open(path, encoding="utf-8"))


def sweep(tiles_dir, seed=SEED, routes=40000, seconds=60.0, batch=2000, use_gpu=True, witness=32,
          cables=None, cables_path=CABLES):
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    cables = cables if cables is not None else load_cables(cables_path)
    rule = {kv: min_radius(cables, kv) for kv in (33, 132)}
    G = ground_from(tiles_dir, xp)
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("kv", "nb", "L2d", "c2", "e3", "p3", "corr", "rel", "nviol", "rmin")}
    meta, samples, intervals, stopped, first = [], 0, 0, "routes", None
    while sum(map(len, cols["kv"])) < routes:
        if cols["kv"] and time.perf_counter() - t0 > seconds:
            stopped = "seconds"; break
        D = draw(rng, min(batch, routes - sum(map(len, cols["kv"]))), G, rule)
        if first is None:
            first = {k: v[:witness].copy() for k, v in D.items()}
        R = pair(D, G)
        samples += R["samples"]; intervals += R["intervals"]
        bend = D["turns"] != 0
        for k, v in (("kv", D["kv"]), ("nb", bend.sum(1)), ("L2d", R["P"]["L2d"]), ("c2", R["c2"]), ("e3", R["e3"]),
                     ("p3", R["p3"]), ("corr", R["corr"]), ("rel", R["rel"]), ("nviol", R["viol"].sum(1)),
                     ("rmin", np.where(bend, R["r"], np.inf).min(1))):
            cols[k].append(v)
        top = np.argsort(-R["rel"])[:KEEP]
        meta += [dict(route_meta(D, R, i, G), rel=float(R["rel"][i])) for i in top]
        if xp is not np:
            cp.cuda.Stream.null.synchronize()
    C = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(C["kv"])
    # CPU witness: the first routes, both channels, NumPy only
    Gc = ground_from(tiles_dir, np)
    Wr = pair(first, Gc)
    wd = np.concatenate([np.abs(Wr["e3"] - C["e3"][:witness]) / C["e3"][:witness],
                         np.abs(Wr["p3"] - C["p3"][:witness]) / C["p3"][:witness]])
    wit = dict(routes=int(len(Wr["e3"])), photons=int((wd > WIT_TOL).sum()), max_rel=float(wd.max()), tol=WIT_TOL)
    photons = int((C["rel"] > TOL).sum())
    plan_rel = np.abs(C["c2"] - C["L2d"]) / C["L2d"]
    q = lambda v: [round(float(x), 6) for x in np.percentile(v, [0, 5, 25, 50, 75, 95, 100])] if len(v) else []
    extra = {}
    for kv in (33, 132):
        m = C["kv"] == kv
        L2, L3 = C["L2d"][m], C["p3"][m]
        extra[f"{kv}kV"] = dict(
            routes=int(m.sum()), plan_km=round(float(L2.sum()) / 1000, 4), cable_3d_km=round(float(L3.sum()) / 1000, 4),
            extra_m_per_km=round(float((L3 - L2).sum() / L2.sum() * 1000), 4),
            extra_pct_percentiles_0_5_25_50_75_95_100=q(100 * (L3 - L2) / L2),
            extra_m_percentiles_0_5_25_50_75_95_100=q(L3 - L2),
            radius_violation_routes=int((C["nviol"][m] > 0).sum()), bent_routes=int((C["nb"][m] > 0).sum()))
    vr = C["nviol"] > 0
    worst = sorted(meta, key=lambda d: -d["rel"])[:KEEP]
    return {
        "kind": "cable length sweep pair", "seed": seed, "device": device,
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "source": {"shape": [G.ny, G.nx], "origin_e_m": G.oe, "origin_n_m": G.on, "spacing_m": G.sp},
        "bend_rules": {f"{kv}kV": rule[kv] for kv in (33, 132)},
        "cables_json_sha256": hashlib.sha256(json.dumps(cables, sort_keys=True).encode()).hexdigest(),
        "method": {"electron": f"3D chords of samples every <= {DS_E} m of chainage on the bilinear ground",
                   "positron": "analytic plan length + vertical correction, pieces cut at grid lines, "
                               "closed form on straights, 6-point Gauss-Legendre on arcs",
                   "tolerance_rel": TOL,
                   "routes_drawn": "0-4 bends uniform; legs log-uniform 3-300 m; turns 5-150 deg either way; "
                                   "33/132 kV half each; each corner asks for the rule radius"},
        "counts": {"routes": n, "by_bends": {str(b): int((C["nb"] == b).sum()) for b in range(NB + 1)},
                   "electron_samples": samples, "positron_intervals": intervals, "stopped_by": stopped},
        "disagreement": {"photons": photons, "max_rel": float(C["rel"].max()), "median_rel": float(np.median(C["rel"])),
                         "worst": worst},
        "plan_check": {"max_rel_chords_vs_analytic": float(plan_rel.max()), "median_rel": float(np.median(plan_rel))},
        "bend_radius": {"routes_with_violation": int(vr.sum()), "bent_routes": int((C["nb"] > 0).sum()),
                        "bends_below_min": int(C["nviol"].sum()), "bends_total": int(C["nb"].sum()),
                        "smallest_fitted_radius_m": float(C["rmin"][np.isfinite(C["rmin"])].min()) if np.isfinite(C["rmin"]).any() else None},
        "extra_3d_over_plan": extra,
        "witness": wit, "photons_total": photons + wit["photons"],
        "bounds": {"routes": routes, "seconds": seconds, "batch": batch},
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def route_meta(D, R, i, G):
    nb = int((D["turns"][i] != 0).sum())
    return dict(start_e=round(float(D["x0"][i] + G.oe), 3), start_n=round(float(D["y0"][i] + G.on), 3),
                heading_deg=round(math.degrees(D["h0"][i]) % 360, 3), kv=int(D["kv"][i]),
                legs_m=[round(float(v), 3) for v in D["legs"][i][:nb + 1]],
                turns_deg=[round(math.degrees(v), 3) for v in D["turns"][i][:nb]],
                fitted_radius_m=[round(float(v), 4) for v in R["r"][i][:nb]],
                plan_m=round(float(R["P"]["L2d"][i]), 4), electron_m=round(float(R["e3"][i]), 6),
                positron_m=round(float(R["p3"][i]), 6))


def write_receipt(tiles_dir, r):
    body = (json.dumps(r, indent=1, sort_keys=True) + "\n").encode()
    path = os.path.join(tiles_dir, RECEIPT)
    with open(path, "wb") as f:       # bytes: LF on every platform
        f.write(body)
    return path, hashlib.sha256(body).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--routes", type=int, default=40000)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--cables", default=CABLES)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    r = sweep(a.tiles, a.seed, a.routes, a.seconds, a.batch, use_gpu=not a.cpu, cables_path=a.cables)
    path, sha = write_receipt(a.tiles, r)
    c, d, b = r["counts"], r["disagreement"], r["bend_radius"]
    print(f"{c['routes']} routes {c['by_bends']} on {r['device']} in {r['wall_s']} s, stopped by {c['stopped_by']}")
    for kv, rr in r["bend_rules"].items():
        print(f"  {kv}: R = {rr['multiple_of_od']} x {rr['od_mm']} mm = {rr['radius_m']:.3f} m, depth {rr['depth_m']} m")
    print(f"photons {d['photons']} (> {TOL:.1%}), max rel {d['max_rel']:.2e}, median {d['median_rel']:.2e}; "
          f"plan chords vs analytic max {r['plan_check']['max_rel_chords_vs_analytic']:.2e}")
    print(f"bends below min radius: {b['bends_below_min']} of {b['bends_total']} in {b['routes_with_violation']} routes")
    for kv, e in r["extra_3d_over_plan"].items():
        print(f"  {kv}: {e['plan_km']} km plan -> {e['cable_3d_km']} km 3D, +{e['extra_m_per_km']} m/km, "
              f"% p50/p95/max {e['extra_pct_percentiles_0_5_25_50_75_95_100'][3:4] + e['extra_pct_percentiles_0_5_25_50_75_95_100'][5:]}")
    print(f"witness {r['witness']['routes']} routes, photons {r['witness']['photons']} (max {r['witness']['max_rel']:.1e})")
    print(f"receipt {path} sha256 {sha}")
    return 0 if r["photons_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
