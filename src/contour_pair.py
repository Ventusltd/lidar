"""Contour segments from a south-up height grid, computed as a pair on the GPU with a CPU witness.

Node (r, c) sits at local x = c, y = r (grid units; multiply by the spacing for metres). A node is ABOVE a
level L when z >= L, so a crossing lies on an edge whose ends are on opposite sides, at t in (0, 1] from the
lower end. Cells with a missing corner (NaN) are skipped by every channel. Edge ids:
    horizontal edge (r, c)-(r, c+1):  r*C + c          vertical edge (r, c)-(r+1, c):  R*C + r*C + c
Cell (r, c) has edges S = H(r, c), E = V(r, c+1), N = H(r+1, c), W = V(r, c); corners a = SW, b = SE,
c = NE, d = NW. Saddles (opposite corners above) are resolved by the cell centre (mean of the corners):
the centre joins the corners that share its side (the asymptotic-decider shortcut of Nielson and Hamann,
1991, as used in marching squares since Lorensen and Cline, 1987).

  electron   marching squares: a 4-bit corner case per cell, a lookup table of edge pairs, crossings
             placed inside the cell kernel by the weighted form (p0 (z1 - L) + p1 (L - z0)) / (z1 - z0).
  positron   crossings found along rows and along columns independently, t = (L - z0) / (z1 - z0), then
             joined per cell from a 4-bit EDGE mask; saddles by "centre sum >= 4L equals corner a".
  photons    segments that one channel has and the other does not, and crossings whose positions differ
             by more than TOL_XY. Counted per level, never suppressed.
  witness    NumPy on the CPU by a third path: np.diff of the above-mask for crossings, the complementary
             form x = (c + 1) - (z1 - L) / (z1 - z0), and the whole segment set rebuilt from edge counts.
"""
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the same arithmetic runs in NumPy and the receipt says so
    cp = None

TOL_XY = 1e-9          # grid units: the photon bound on a crossing position
S_, E_, N_, W_ = 0, 1, 2, 3

# electron table: corner case (a=1, b=2, c=4, d=8) + 16 * centre_above -> two edge pairs (or -1)
_PAIRS = {1: [(W_, S_)], 2: [(S_, E_)], 3: [(W_, E_)], 4: [(E_, N_)], 6: [(S_, N_)], 7: [(W_, N_)],
          8: [(N_, W_)], 9: [(S_, N_)], 11: [(E_, N_)], 12: [(W_, E_)], 13: [(S_, E_)], 14: [(W_, S_)],
          5: [(W_, S_), (E_, N_)], 21: [(S_, E_), (N_, W_)],     # a, c above: centre below / above
          10: [(S_, E_), (N_, W_)], 26: [(W_, S_), (E_, N_)]}    # b, d above: centre below / above
TABLE = np.full((32, 2, 2), -1, np.int8)
for _k, _v in _PAIRS.items():
    for _slot, _p in enumerate(_v):
        TABLE[_k, _slot] = _p
        if _k not in (5, 10, 21, 26):
            TABLE[_k + 16, _slot] = _p      # centre matters only at saddles
# positron table: edge mask (S=1, E=2, N=4, W=8) with exactly two bits -> the pair
EDGE_PAIR = np.full((16, 2), -1, np.int8)
for _m in range(16):
    _bits = [k for k in range(4) if _m >> k & 1]
    if len(_bits) == 2:
        EDGE_PAIR[_m] = _bits


def corners(z):
    return z[:-1, :-1], z[:-1, 1:], z[1:, 1:], z[1:, :-1]


def cell_edge_ids(xp, rr, cc, R, C):
    return xp.stack([rr * C + cc, R * C + rr * C + cc + 1, (rr + 1) * C + cc, R * C + rr * C + cc], 1)


def electron(xp, z, L):
    """Marching squares. Returns cell rows, cols, edge ids A, B and positions xa, ya, xb, yb."""
    R, C = z.shape
    a, b, c, d = corners(z)
    ok = xp.isfinite(a) & xp.isfinite(b) & xp.isfinite(c) & xp.isfinite(d)
    case = ((a >= L).astype(xp.int16) + 2 * (b >= L) + 4 * (c >= L) + 8 * (d >= L))
    rr, cc = xp.nonzero(ok & (case != 0) & (case != 15))
    A, B, Cz, D = a[rr, cc], b[rr, cc], c[rr, cc], d[rr, cc]
    key = case[rr, cc] + 16 * ((A + B + Cz + D) / 4 >= L).astype(xp.int16)
    x0, y0 = cc.astype(xp.float64), rr.astype(xp.float64)
    x1, y1 = x0 + 1, y0 + 1
    wx = lambda z0, z1, p0, p1: (p0 * (z1 - L) + p1 * (L - z0)) / (z1 - z0)
    ex = xp.stack([wx(A, B, x0, x1), x1, wx(D, Cz, x0, x1), x0], 1)
    ey = xp.stack([y0, wx(B, Cz, y0, y1), y1, wx(A, D, y0, y1)], 1)
    eid = cell_edge_ids(xp, rr.astype(xp.int64), cc.astype(xp.int64), R, C)
    codes = xp.asarray(TABLE)[key]
    out = []
    for slot in (0, 1):
        pa, pb = codes[:, slot, 0].astype(xp.int64), codes[:, slot, 1].astype(xp.int64)
        sel = xp.nonzero(pa >= 0)[0]
        pa, pb = pa[sel], pb[sel]
        take = lambda arr, p: arr[sel, p]
        out.append((rr[sel], cc[sel], take(eid, pa), take(eid, pb), take(ex, pa), take(ey, pa),
                    take(ex, pb), take(ey, pb)))
    return tuple(xp.concatenate([o[k] for o in out]) for k in range(8)), int((((case == 5) | (case == 10)) & ok).sum())


def positron(xp, z, L):
    """Row and column crossings found independently, then joined. Returns (ids, x, y), (idA, idB), saddles, odd."""
    R, C = z.shape
    fin, up = xp.isfinite(z), z >= L
    sH = fin[:, :-1] & fin[:, 1:] & (up[:, :-1] != up[:, 1:])
    sV = fin[:-1, :] & fin[1:, :] & (up[:-1, :] != up[1:, :])
    rH, cH = xp.nonzero(sH)
    z0, z1 = z[rH, cH], z[rH, cH + 1]
    hx, hy = cH + (L - z0) / (z1 - z0), rH.astype(xp.float64)
    rV, cV = xp.nonzero(sV)
    z0, z1 = z[rV, cV], z[rV + 1, cV]
    vx, vy = cV.astype(xp.float64), rV + (L - z0) / (z1 - z0)
    ids = xp.concatenate([rH.astype(xp.int64) * C + cH, R * C + rV.astype(xp.int64) * C + cV])
    X, Y = xp.concatenate([hx, vx]), xp.concatenate([hy, vy])
    a, b, c, d = corners(z)
    ok = fin[:-1, :-1] & fin[:-1, 1:] & fin[1:, 1:] & fin[1:, :-1]
    m = (sH[:-1, :].astype(xp.int16) + 2 * sV[:, 1:] + 4 * sH[1:, :] + 8 * sV[:, :-1]) * ok
    pop = (m & 1) + (m >> 1 & 1) + (m >> 2 & 1) + (m >> 3 & 1)
    odd = int(((pop == 1) | (pop == 3)).sum())
    r2, c2 = xp.nonzero(pop == 2)
    eid2 = cell_edge_ids(xp, r2.astype(xp.int64), c2.astype(xp.int64), R, C)
    pr = xp.asarray(EDGE_PAIR)[m[r2, c2]].astype(xp.int64)
    ar = xp.arange(r2.size)
    ia, ib = [eid2[ar, pr[:, 0]]], [eid2[ar, pr[:, 1]]]
    r4, c4 = xp.nonzero(pop == 4)
    eid4 = cell_edge_ids(xp, r4.astype(xp.int64), c4.astype(xp.int64), R, C)
    same = (a[r4, c4] >= L) == (a[r4, c4] + b[r4, c4] + c[r4, c4] + d[r4, c4] >= 4 * L)
    # same: centre joins a, so b and d are cut off -> (S, E), (N, W); else a and c -> (W, S), (E, N)
    ia += [xp.where(same, eid4[:, S_], eid4[:, W_]), xp.where(same, eid4[:, N_], eid4[:, E_])]
    ib += [xp.where(same, eid4[:, E_], eid4[:, S_]), xp.where(same, eid4[:, W_], eid4[:, N_])]
    return (ids, X, Y), (xp.concatenate(ia), xp.concatenate(ib)), int(r4.size), odd


def seg_keys(xp, ia, ib, ne):
    return xp.minimum(ia, ib) * ne + xp.maximum(ia, ib)


def set_diff_count(xp, k1, k2):
    """Keys in one set and not the other (both sets are duplicate-free)."""
    common = k1.size + k2.size - xp.unique(xp.concatenate([k1, k2])).size
    return int(k1.size - common), int(k2.size - common)


def lookup(xp, ids, vals, want):
    """vals at want (ids sorted); returns values and a found mask."""
    if ids.size == 0:
        return xp.zeros(want.shape), xp.zeros(want.shape, bool)
    i = xp.clip(xp.searchsorted(ids, want), 0, ids.size - 1)
    return vals[i], ids[i] == want


def pair_level(xp, z, L):
    """Both channels at one level. Returns electron segments (on the device) and the tally."""
    R, C = z.shape
    ne = 2 * R * C
    el, el_saddles = electron(xp, z, L)
    (ids, X, Y), (pa, pb), po_saddles, odd = positron(xp, z, L)
    only_e, only_p = set_diff_count(xp, seg_keys(xp, el[2], el[3], ne), seg_keys(xp, pa, pb, ne))
    worst, missing = 0.0, 0
    for eid, ex, ey in ((el[2], el[4], el[5]), (el[3], el[6], el[7])):
        px, f1 = lookup(xp, ids, X, eid)
        py, _ = lookup(xp, ids, Y, eid)
        missing += int((~f1).sum())
        if bool(f1.any()):
            dd = xp.maximum(xp.abs(px - ex), xp.abs(py - ey))[f1]
            worst = max(worst, float(dd.max()))
    tally = dict(segments_e=int(el[0].size), segments_p=int(pa.size), only_e=only_e, only_p=only_p,
                 saddles_e=el_saddles, saddles_p=po_saddles, odd_cells=odd, crossings=int(ids.size),
                 unmatched_ends=missing, max_dxy=worst)
    return el, tally


def witness_level(z, L, seg):
    """CPU witness at one level against the electron segments (host arrays). Returns a photon tally."""
    R, C = z.shape
    ne = 2 * R * C
    fin = np.isfinite(z)
    up = (z >= L).astype(np.int8)
    hc = (np.diff(up, axis=1) != 0) & fin[:, :-1] & fin[:, 1:]
    vc = (np.diff(up, axis=0) != 0) & fin[:-1, :] & fin[1:, :]
    rH, cH = np.nonzero(hc); rV, cV = np.nonzero(vc)
    zh0, zh1, zv0, zv1 = z[rH, cH], z[rH, cH + 1], z[rV, cV], z[rV + 1, cV]
    wid = np.concatenate([rH.astype(np.int64) * C + cH, R * C + rV.astype(np.int64) * C + cV])
    wx = np.concatenate([(cH + 1) - (zh1 - L) / (zh1 - zh0), cV.astype(np.float64)])
    wy = np.concatenate([rH.astype(np.float64), (rV + 1) - (zv1 - L) / (zv1 - zv0)])
    # the segment set rebuilt from the four edge flags of every whole cell
    ok = fin[:-1, :-1] & fin[:-1, 1:] & fin[1:, 1:] & fin[1:, :-1]
    flags = np.stack([hc[:-1, :], vc[:, 1:], hc[1:, :], vc[:, :-1]], -1) & ok[..., None]
    cnt = flags.sum(-1)
    r2, c2 = np.nonzero(cnt == 2)
    e2 = np.asarray(cell_edge_ids(np, r2.astype(np.int64), c2.astype(np.int64), R, C))
    pick = np.sort(np.where(flags[r2, c2], e2, np.iinfo(np.int64).max), 1)[:, :2]
    keys = [pick[:, 0] * ne + pick[:, 1]]
    r4, c4 = np.nonzero(cnt == 4)
    e4 = np.asarray(cell_edge_ids(np, r4.astype(np.int64), c4.astype(np.int64), R, C))
    cen = np.mean(np.stack([z[r4, c4], z[r4, c4 + 1], z[r4 + 1, c4 + 1], z[r4 + 1, c4]]), 0) >= L
    same = cen == (z[r4, c4] >= L)
    for p, q in (((S_, E_), (W_, S_)), ((N_, W_), (E_, N_))):
        u = np.where(same, e4[:, p[0]], e4[:, q[0]]); v = np.where(same, e4[:, p[1]], e4[:, q[1]])
        keys.append(np.minimum(u, v) * ne + np.maximum(u, v))
    wkeys = np.concatenate(keys)
    only_w, only_e = set_diff_count(np, wkeys, seg_keys(np, seg[2], seg[3], ne))
    worst, missing = 0.0, 0
    for eid, ex, ey in ((seg[2], seg[4], seg[5]), (seg[3], seg[6], seg[7])):
        order = np.argsort(wid)
        px, f1 = lookup(np, wid[order], wx[order], eid)
        py, _ = lookup(np, wid[order], wy[order], eid)
        missing += int((~f1).sum())
        if f1.any():
            worst = max(worst, float(np.maximum(np.abs(px - ex), np.abs(py - ey))[f1].max()))
    far = worst > TOL_XY
    return dict(segments=int(wkeys.size), only_w=only_w, only_e=only_e, saddles=int(r4.size),
                unmatched_ends=missing, max_dxy=worst,
                photons=only_w + only_e + missing + int(far))
