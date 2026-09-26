# SPDX-License-Identifier: Apache-2.0
"""Ground and plan geometry for cable_sweep.py: bilinear ground, the bend rule, filleted plan pieces.

Split out of cable_sweep.py to keep each script under 400 lines; see cable_sweep.py for the method.
"""
import json, os
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card, no CuPy: the same arithmetic runs in NumPy and says so
    cp = None


NB = 4                # most bends a route has
NP = 2 * NB + 1       # pieces a route: line, arc, line, ... line
CSA = 1000
DEPTH = {33: 0.9, 132: 1.05}


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
