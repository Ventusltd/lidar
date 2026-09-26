# SPDX-License-Identifier: Apache-2.0
"""Keep simplified contours from crossing: check every tile and put dropped vertices back until none cross.

Douglas-Peucker runs on each line alone, so two simplified lines can cross even though the traced
(marching-squares) lines cannot (tester 3 found 20 crossing pairs on open-land-01, 3 between levels). After
simplifying, every tile is checked as it will be stored: vertices in whole centimetres from the tile's
south-west corner, integer orientation tests, so a crossing is decided exactly. Any two segments that
touch or cross count, except neighbours on one line sharing their vertex. For each crossing, each of the two
spans involved gets back its dropped vertex furthest from the span; this repeats until nothing crosses or
nothing is left to put back (then it is COUNTED in the receipt, never hidden; a touch left by the
centimetre rounding of the traced line itself is counted apart from a crossing). Putting a vertex
back only brings a span closer to the traced line, so the Douglas-Peucker tolerance still holds.
"""
import numpy as np

MAX_ROUNDS = 64


def _dist(p, a, b):
    d = b - a
    L2 = float(d @ d)
    if L2 == 0.0:
        return np.hypot(*(p - a).T)
    return np.abs(d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])) / np.sqrt(L2)


def _segments(lines, x0, y0):
    """All stored segments of a tile: int64 cm ends, and (line, kept i, kept j, last index) of each span."""
    P, Q, own = [], [], []
    for n, (p, keep) in enumerate(lines):
        kept = np.nonzero(keep)[0]
        if len(kept) < 2:
            continue
        cm = np.rint((p[kept] - (x0, y0)) * 100).astype(np.int64)
        P.append(cm[:-1]); Q.append(cm[1:])
        own.append(np.stack([np.full(len(kept) - 1, n), kept[:-1], kept[1:], np.full(len(kept) - 1, len(p) - 1)], 1))
    if not P:
        z = np.zeros((0, 2), np.int64)
        return z, z, np.zeros((0, 4), np.int64)
    return np.concatenate(P), np.concatenate(Q), np.concatenate(own)


def _orient(a, b, c):
    return np.sign((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _on(a, b, c):
    """c on the box of a-b (used when collinear)."""
    return ((np.minimum(a[:, 0], b[:, 0]) <= c[:, 0]) & (c[:, 0] <= np.maximum(a[:, 0], b[:, 0]))
            & (np.minimum(a[:, 1], b[:, 1]) <= c[:, 1]) & (c[:, 1] <= np.maximum(a[:, 1], b[:, 1])))


def crossings(P, Q, own):
    """Rows (s, t, proper) of segments that touch or cross, bar same-line neighbours sharing a vertex.
    proper = 1 where they pass through each other; 0 where they only touch (usually two crossings of the
    traced line within half a centimetre of each other, rounded to one point)."""
    n = len(P)
    if n < 2:
        return np.zeros((0, 3), np.int64)
    lo, hi = np.minimum(P, Q), np.maximum(P, Q)
    order = np.argsort(lo[:, 0], kind="stable")
    xs = lo[order, 0]
    ends = np.searchsorted(xs, hi[order, 0], side="right")
    cnt = ends - np.arange(n) - 1
    s = np.repeat(np.arange(n), cnt)
    t = s + 1 + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    s, t = order[s], order[t]
    ok = (lo[s, 1] <= hi[t, 1]) & (lo[t, 1] <= hi[s, 1])
    s, t = s[ok], t[ok]
    same = own[s, 0] == own[t, 0]
    nb = same & ((own[s, 2] == own[t, 1]) | (own[t, 2] == own[s, 1]))
    ring = same & (((own[s, 1] == 0) & (P[s] == Q[t]).all(1)) | ((own[t, 1] == 0) & (P[t] == Q[s]).all(1)))
    s, t = s[~nb & ~ring], t[~nb & ~ring]
    a, b, c, d = P[s], Q[s], P[t], Q[t]
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    hit = (o1 * o2 < 0) & (o3 * o4 < 0)
    proper = (o1 * o2 < 0) & (o3 * o4 < 0)
    hit |= (o1 == 0) & _on(a, b, c) | (o2 == 0) & _on(a, b, d) | (o3 == 0) & _on(c, d, a) | (o4 == 0) & _on(c, d, b)
    # two open lines ending at the same tile-edge point (their crossings round to one centimetre) only touch
    s0, s1 = own[s, 1] == 0, own[s, 2] == own[s, 3]
    t0, t1 = own[t, 1] == 0, own[t, 2] == own[t, 3]
    eq = lambda u, v: (u == v).all(1)
    ends = ((eq(a, c) & s0 & t0) | (eq(a, d) & s0 & t1) | (eq(b, c) & s1 & t0) | (eq(b, d) & s1 & t1)) & (own[s, 0] != own[t, 0])
    hit &= proper | ~ends
    return np.stack([s[hit], t[hit], proper[hit]], 1)


def untangle(lines, x0, y0):
    """lines: [(p (n,2) metres, keep bool (n,))], keep masks edited in place.
    Returns (crossings found, crossings left, touches left, vertices restored)."""
    found = restored = left = touch = 0
    for rnd in range(MAX_ROUNDS):
        P, Q, own = _segments(lines, x0, y0)
        hits = crossings(P, Q, own)
        if rnd == 0:
            found = int(hits[:, 2].sum())
        left, touch = int(hits[:, 2].sum()), int((hits[:, 2] == 0).sum())
        if not len(hits):
            break
        added = 0
        for n, i, j, _ in {tuple(own[k]) for k in hits[:, :2].ravel()}:
            if j <= i + 1:
                continue
            p, keep = lines[n]
            dv = _dist(p[i + 1:j], p[i], p[j])
            keep[i + 1 + int(np.argmax(dv))] = True
            added += 1
        restored += added
        if not added:
            break
    return found, left, touch, restored
