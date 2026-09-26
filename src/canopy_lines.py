"""Hedgerow lines from a vegetation mask: open, thin, trace, keep the long narrow runs.

Vegetation that survives a square opening of half-width OPEN_R (a (2r+1) m square fits inside it) is
wide: woodland, copses, big single crowns. What the opening removes is narrow. Narrow vegetation is
thinned to a one-cell skeleton (Zhang and Suen, 1984, "A fast parallel algorithm for thinning digital
patterns", CACM 27(3):236-239), the skeleton is traced into polylines, polylines are simplified
(Douglas-Peucker, tolerance SIMPLIFY_M) and only runs of at least MIN_LEN_M are kept as hedgerows.
Narrow cells within HEDGE_R of a kept run are hedge; every other vegetation cell is tree canopy.
A 1 m closing first bridges single-cell gaps so a hedge with a thin spot is still one run.

Known limits, stated rather than hidden: a hedgerow tree whose crown is wider than 2*OPEN_R+1 m is
wide, so it is drawn as a tree and the hedge line breaks there; a narrow strip of woodland edge (a
fringe under 2*OPEN_R+1 m) can pass as hedge; a line of separate trees closer than about 1 m reads as
one hedge. Pure NumPy; everything is on the 1 m node grid, rows south to north.
"""
import numpy as np

OPEN_R = 4          # opening half-width, m: vegetation narrower than 9 m is "narrow"
HEDGE_R = 3         # narrow cells within 3 m (chessboard) of a kept run are hedge
MIN_LEN_M = 15.0    # a hedgerow is at least this long
SIMPLIFY_M = 0.75   # Douglas-Peucker tolerance
FRINGE_M = 2        # narrow cells this close to wide vegetation are the wood's edge, not hedge
JOIN_M = 7.0        # skeleton ends this close are one hedge for the length test: a 4-5 m gate plus
                    # the metre or so thinning takes off each end
SPUR_M = 3.0        # skeleton pieces shorter than this inside a kept run are spurs
DEPTH_MAX = 8       # erosion depth cap when measuring half-widths

N8 = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def shift(m, dr, dc, fill=False):
    """m moved so out[r, c] = m[r - dr, c - dc]; cells from outside the grid take fill."""
    out = np.full_like(m, fill)
    R, C = m.shape
    out[max(dr, 0):R + min(dr, 0), max(dc, 0):C + min(dc, 0)] = \
        m[max(-dr, 0):R + min(-dr, 0), max(-dc, 0):C + min(-dc, 0)]
    return out


def erode(m, r=1):
    """Square erosion of half-width r (outside the grid counts as empty)."""
    out = m.copy()
    for _ in range(r):
        nxt = out.copy()
        for dr, dc in N8:
            nxt &= shift(out, dr, dc, False)
        out = nxt
    return out


def dilate(m, r=1):
    out = m.copy()
    for _ in range(r):
        nxt = out.copy()
        for dr, dc in N8:
            nxt |= shift(out, dr, dc, False)
        out = nxt
    return out


def opening(m, r=OPEN_R):
    return dilate(erode(m, r), r)


def opening_witness(m, r=OPEN_R):
    """The same opening by one (2r+1)-square window pass each way (sliding windows, not shifts)."""
    k = 2 * r + 1
    W = np.lib.stride_tricks.sliding_window_view
    e = W(np.pad(m, r, constant_values=False), (k, k)).all((-1, -2))
    return W(np.pad(e, r, constant_values=False), (k, k)).any((-1, -2))


def depth(m, cap=DEPTH_MAX):
    """How many unit erosions each cell survives (0 = on the edge), capped."""
    d = np.zeros(m.shape, np.uint8)
    cur = m.copy()
    for k in range(1, cap + 1):
        cur = erode(cur, 1)
        if not cur.any():
            break
        d[cur] = k
    return d


def thin(m):
    """Zhang-Suen thinning to an 8-connected skeleton one cell wide."""
    img = m.copy()
    while True:
        changed = False
        for step in (0, 1):
            # P2..P9 clockwise from north; rows run south to north, so north is +1 row.
            p = [shift(img, -dr, -dc, False) for dr, dc in
                 ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))]
            b = sum(x.astype(np.uint8) for x in p)
            a = sum(((~p[i]) & p[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            p2, p4, p6, p8 = p[0], p[2], p[4], p[6]
            if step == 0:
                c1, c2 = ~(p2 & p4 & p6), ~(p4 & p6 & p8)
            else:
                c1, c2 = ~(p2 & p4 & p8), ~(p2 & p6 & p8)
            kill = img & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if kill.any():
                img &= ~kill
                changed = True
        if not changed:
            return img


def trace(skel):
    """Skeleton -> list of polylines [(r, c), ...]. Nodes are cells with other than two neighbours."""
    rows, cols = np.nonzero(skel)
    on = set(zip(rows.tolist(), cols.tolist()))
    nb = {p: [(p[0] + dr, p[1] + dc) for dr, dc in N8 if (p[0] + dr, p[1] + dc) in on] for p in on}
    node = {p for p, v in nb.items() if len(v) != 2}
    seen, lines = set(), []

    def walk(a, b):
        line = [a, b]
        seen.add((a, b)); seen.add((b, a))
        prev, cur = a, b
        while cur not in node:
            nxt = [q for q in nb[cur] if q != prev and (cur, q) not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add((prev, cur)); seen.add((cur, prev))
            line.append(cur)
        return line

    for p in sorted(node):
        for q in nb[p]:
            if (p, q) not in seen:
                lines.append(walk(p, q))
    for p in sorted(on):                      # closed loops with no node
        for q in nb[p]:
            if (p, q) not in seen:
                node.add(p)
                lines.append(walk(p, q))
                node.discard(p)
    return [l for l in lines if len(l) >= 2]


def path_len(pts):
    a = np.asarray(pts, float)
    return float(np.hypot(*np.diff(a, axis=0).T).sum()) if len(a) > 1 else 0.0


def simplify(pts, tol=SIMPLIFY_M):
    """Douglas-Peucker on a list of (r, c)."""
    a = np.asarray(pts, float)
    if len(a) < 3:
        return [tuple(map(int, p)) for p in a]
    keep = np.zeros(len(a), bool); keep[0] = keep[-1] = True
    stack = [(0, len(a) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = a[j] - a[i]; L = np.hypot(*d)
        seg = a[i + 1:j] - a[i]
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k; keep[m] = True
            stack += [(i, m), (m, j)]
    return [tuple(map(int, p)) for p in a[keep]]


def components(lines, join=0.0):
    """Group traced polylines that share a cell (8-connected skeleton pieces) -> list of index lists.

    join > 0 also groups pieces whose ends lie within join metres (a hedge broken by a field gate).
    """
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, l in enumerate(lines):
        for p in (l[0], l[-1]):
            j = owner.setdefault(p, i)
            if j != i:
                parent[find(i)] = find(j)
    if join > 0 and lines:
        ends = np.asarray([p for l in lines for p in (l[0], l[-1])], float)
        who = np.repeat(np.arange(len(lines)), 2)
        for a in range(len(ends)):
            d = np.hypot(*(ends[a + 1:] - ends[a]).T)
            for b in np.nonzero(d <= join)[0] + a + 1:
                parent[find(int(who[a]))] = find(int(who[b]))
    groups = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def hedgerows(veg, chm, oe=0, on=0):
    """veg: bool vegetation mask; chm: canopy height (m). Returns (hedge mask, wide mask, lines, stats).

    Narrow = the closed mask minus the opened mask grown by FRINGE_M, so the ragged edge of a wood is
    not taken for hedge. A skeleton piece is kept when its connected run spans at least MIN_LEN_M
    (bounding-box diagonal), then every piece of that run longer than SPUR_M is a polyline.
    lines: [{id, pts: [[e, n, h], ...], length_m, width_m, mean_h, max_h}], heights the 3x3 maximum.
    """
    closed = erode(dilate(veg, 1), 1) | veg
    wide = opening(veg, OPEN_R) & veg
    narrow = closed & ~dilate(opening(closed, OPEN_R), FRINGE_M)
    skel = thin(narrow)
    dep = depth(narrow)
    h0 = np.where(np.isfinite(chm) & veg, chm, 0.0)
    hmax = h0.copy()
    for dr, dc in N8:
        hmax = np.maximum(hmax, shift(h0, dr, dc, 0.0))
    pieces = trace(skel)
    keep_px = np.zeros(veg.shape, bool)
    lines, runs, dropped = [], 0, 0
    for group in components(pieces, JOIN_M):
        cells = np.asarray([p for i in group for p in pieces[i]])
        span = float(np.hypot(*(cells.max(0) - cells.min(0))))
        if span < MIN_LEN_M:
            dropped += 1
            continue
        runs += 1
        for i in group:
            raw = pieces[i]
            L = path_len(raw)
            if L < SPUR_M:
                continue
            r, c = np.asarray(raw).T
            keep_px[r, c] = True
            pts = simplify(raw)
            lines.append(dict(id=len(lines), run=runs - 1,
                              pts=[[int(oe + p[1]), int(on + p[0]), round(float(hmax[p]), 1)] for p in pts],
                              length_m=round(L, 1), width_m=round(float(2 * dep[r, c].mean() + 1), 1),
                              mean_h=round(float(hmax[r, c].mean()), 2), max_h=round(float(hmax[r, c].max()), 2)))
    hedge = veg & narrow & dilate(keep_px, HEDGE_R)
    stats = dict(open_r_m=OPEN_R, fringe_m=FRINGE_M, join_m=JOIN_M, hedge_r_m=HEDGE_R, min_len_m=MIN_LEN_M, spur_m=SPUR_M,
                 simplify_m=SIMPLIFY_M, narrow_cells=int(narrow.sum()), wide_cells=int(wide.sum()),
                 skeleton_cells=int(skel.sum()), runs_kept=runs, runs_too_short=dropped, polylines=len(lines),
                 hedge_km=round(sum(l["length_m"] for l in lines) / 1000, 3))
    return hedge, wide, lines, stats
