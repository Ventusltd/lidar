"""The two viewshed methods in NumPy: the CPU witness for src/viewshed.py, and its no-card fallback.

Both methods are written here as whole-array NumPy (every target cell, or every ray, at once), which is a
different code path from the one-thread-per-cell CUDA kernels in viewshed.py. The arithmetic is float32 with
the operations in the same order as the kernels (compiled with FMA contraction off), so the two agree to the
bit: any difference is a fault, not rounding, and the receipt counts it.

Geometry shared by both (Franklin and Ray, 1994, see viewshed.py for the full citation):
  * heights sit on grid nodes; the observer stands on a node;
  * a sight line from the observer node O to a node T is walked one node step at a time along its MAJOR
    axis (the axis with the larger offset; columns when the offsets are equal). At step k of n the line
    crosses a whole row or column exactly, and the height there is interpolated linearly between the two
    nodes either side on the MINOR axis;
  * the elevation "slope" of a point at distance d is (h - c*d*d - z_eye) / d, where c*d*d is the drop for
    Earth curvature less refraction. A target is visible when its own slope is at least the steepest slope
    met before it (ties count as visible).

R3 (electron): every target gets its own exact sight line, n - 1 samples.
R2 (positron): one ray from O to each node on the border of the grid; at each step the nearest node to
the ray is judged against the ray's running horizon. A node is visible if ANY ray that passes it says so.
"""
import numpy as np

F = np.float32
NODATA = 255


def perimeter(rows, cols):
    """Every border node once, as an (m, 2) int array of (row, col)."""
    top = [(0, c) for c in range(cols)]
    bottom = [(rows - 1, c) for c in range(cols)] if rows > 1 else []
    sides = [(r, c) for r in range(1, rows - 1) for c in ((0, cols - 1) if cols > 1 else (0,))]
    return np.array(top + bottom + sides, dtype=np.int32).reshape(-1, 2)


def sample(z, ro, co, dr, dc, k, n):
    """Height where the line (dr, dc) from (ro, co) crosses major step k of n.

    dr, dc, n are int arrays, k a scalar or array. Returns (h, base row, base col, rem, col_major):
    the node at (base row, base col) is the lower of the two minor-axis neighbours; rem/n is the fraction."""
    colmaj = np.abs(dc) >= np.abs(dr)
    minor = np.where(colmaj, dr, dc) * k
    fl = minor // n                          # floor division, as fdiv() in the kernel
    rem = minor - fl * n
    step = np.where(colmaj, np.sign(dc), np.sign(dr)) * k
    r0 = np.where(colmaj, ro + fl, ro + step)
    c0 = np.where(colmaj, co + step, co + fl)
    r1 = np.where(colmaj & (rem > 0), r0 + 1, r0)
    c1 = np.where(~colmaj & (rem > 0), c0 + 1, c0)
    z0, z1 = z[r0, c0], z[r1, c1]
    f = rem.astype(F) / n.astype(F)
    h = np.where(rem == 0, z0, z0 + f * (z1 - z0))
    return h, r0, c0, rem, colmaj


def _slopes(h, d, zo, curv):
    return (h - curv * d * d - zo) / d


def r3(z, sp, ro, co, eye, tgt, curv):
    """Exact sight line to every node. z: float32 south-up grid (NaN = no data). Returns u8 0/1/255."""
    z = np.asarray(z, F)
    rows, cols = z.shape
    sp, eye, tgt, curv = F(sp), F(eye), F(tgt), F(curv)
    zo = z[ro, co] + eye
    rr, cc = np.mgrid[0:rows, 0:cols]
    dr, dc = (rr - ro).ravel(), (cc - co).ravel()
    n = np.maximum(np.abs(dr), np.abs(dc))
    order = np.argsort(n, kind="stable")    # targets needing more samples sit at the end
    dr, dc, n = dr[order], dc[order], n[order]
    smax = np.full(n.shape, -np.inf, F)
    for k in range(1, int(n.max()) if n.size else 0):
        a = np.searchsorted(n, k, side="right")          # targets with n > k still have a sample at k
        h, *_ = sample(z, ro, co, dr[a:], dc[a:], k, n[a:])
        d = D_of(dr[a:], dc[a:], sp) * F(k) / n[a:].astype(F)
        smax[a:] = np.fmax(smax[a:], _slopes(h, d, zo, curv))
    D = D_of(dr, dc, sp)
    zt = z[ro + dr, co + dc]
    with np.errstate(divide="ignore", invalid="ignore"):
        st = (zt + tgt - curv * D * D - zo) / D
    vis = np.where(n == 0, 1, (st >= smax).astype(np.uint8)).astype(np.uint8)
    out = np.empty(rows * cols, np.uint8)
    out[order] = vis
    out = out.reshape(rows, cols)
    out[np.isnan(z)] = NODATA
    return out


def D_of(dr, dc, sp):
    """Horizontal distance in metres, as the kernels compute it: sp * sqrtf((float)(dr*dr + dc*dc))."""
    return F(sp) * np.sqrt((dr * dr + dc * dc).astype(F))


def r2(z, sp, ro, co, eye, tgt, curv, per=None):
    """Radial sweep: a ray to every border node, nearest node judged at each step. Returns u8 0/1/255."""
    z = np.asarray(z, F)
    rows, cols = z.shape
    sp, eye, tgt, curv = F(sp), F(eye), F(tgt), F(curv)
    zo = z[ro, co] + eye
    per = perimeter(rows, cols) if per is None else per
    dr, dc = per[:, 0] - ro, per[:, 1] - co
    n = np.maximum(np.abs(dr), np.abs(dc))
    keep = n > 0
    dr, dc, n = dr[keep], dc[keep], n[keep]
    DP = D_of(dr, dc, sp)
    out = np.zeros((rows, cols), np.uint8)
    out[ro, co] = 1
    smax = np.full(n.shape, -np.inf, F)
    for k in range(1, int(n.max()) + 1 if n.size else 1):
        a = k <= n
        h, r0, c0, rem, colmaj = sample(z, ro, co, dr[a], dc[a], k, n[a])
        up = (2 * rem >= n[a]).astype(np.int32)
        tr = np.where(colmaj, r0 + up, r0)
        tc = np.where(colmaj, c0, c0 + up)
        er, ec = tr - ro, tc - co
        Dc = D_of(er, ec, sp)
        st = (z[tr, tc] + tgt - curv * Dc * Dc - zo) / Dc
        seen = st >= smax[a]
        out[tr[seen], tc[seen]] = 1
        d = DP[a] * F(k) / n[a].astype(F)
        smax[a] = np.fmax(smax[a], _slopes(h, d, zo, curv))
    out[np.isnan(z)] = NODATA
    return out
