"""Contour lines at 0.5 m, 1 m and 5 m intervals from a south-up 1 m DTM, paired on the GPU, cut into 256 m tiles.

Every level at 0.5 m is traced (the 1 m and 5 m sets are subsets: a level belongs to an interval when it
is a whole multiple of it). The tracing is a pair (src/contour_pair.py): electron = marching squares on
the grid, positron = crossings found along rows and columns independently then joined; disagreements are
counted per level, and a CPU witness rebuilds every level by a third path.

Segments are chained into polylines inside each 256 m tile (a line that leaves a tile ends on the tile
edge, at the same crossing the neighbour starts from), then simplified by Douglas-Peucker (Douglas and
Peucker, 1973) with a tolerance of DP_TOL_M = 0.25 m, a quarter of the grid spacing and well inside the
horizontal uncertainty that the survey's +-15 cm vertical RMSE implies on gentle ground. The simplified
lines are re-checked: every dropped vertex must lie within the tolerance of its span (max_dp_dev_m). The
tolerance is measured from the traced line, not the true DTM contour (see TOL_BASIS). Simplifying each line
alone can make lines cross, so each tile is then checked as stored and dropped vertices are put back until no
two lines touch or cross (contour_topo.py); any left are counted as photons.
Vertices are then stored in whole centimetres from the tile's south-west corner (a further <= 7.1 mm).

Tile file (ggc1 JSON, LF): {"format": "ggc1", "e0", "n0", "tile_m": 256, "unit_m": 0.01, "dp_tol_m",
  "levels": [{"z": metres, "lines": [[x0, y0, x1, y1, ...], ...]}, ...]}   x, y integer centimetres.
Index contour-tiles.json lists every tile with its sha256; the receipt contour_receipt.json sits beside it.

    python src/contour_tiles.py --site $LIDAR_OUT/open-land-01
"""
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
import numpy as np
from cut_tiles import ea_notice, TERRAIN_ONLY  # noqa: F401
import contour_pair as cpair
import contour_topo as topo_mod

cp = cpair.cp
TILE_M = 256
STEP_M = 0.5
INTERVALS = (0.5, 1.0, 5.0)
DP_TOL_M = 0.25
TOL_BASIS = ("dp_tol_m is the distance from the traced marching-squares line, not from the true contour of the "
             "DTM; in cells near saddles the traced chords themselves depart from it by up to about 0.3 m, so a "
             "stored line can lie up to about 0.55 m from the true contour (tester 3 measured 0.553 m on open-land-01)")
INDEX = "contour-tiles.json"
RECEIPT = "contour_receipt.json"


def levels_for(grid):
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    return list(range(int(np.ceil(lo / STEP_M)), int(np.floor(hi / STEP_M)) + 1))  # level = k * STEP_M


def in_interval(k, iv):
    return (k * STEP_M) % iv == 0


# ---------------------------------------------------------------- chaining within tiles
def chain(tile, ia, ib, xa, ya, xb, yb, ne):
    """Join segments sharing an edge id inside the same tile. Returns [(tile, [(x, y), ...])]."""
    tile, ia, ib = tile.tolist(), ia.tolist(), ib.tolist()
    pos = {}
    for i, x, y in zip(ia, xa.tolist(), ya.tolist()):
        pos[i] = (x, y)
    for i, x, y in zip(ib, xb.tolist(), yb.tolist()):
        pos[i] = (x, y)
    adj = {}
    for s, (t, a, b) in enumerate(zip(tile, ia, ib)):
        adj.setdefault(t * ne + a, []).append(s)
        adj.setdefault(t * ne + b, []).append(s)
    used = bytearray(len(ia))
    out = []

    def walk(t, end):
        got = []
        while True:
            nxt = next((s for s in adj[t * ne + end] if not used[s]), None)
            if nxt is None:
                return got
            used[nxt] = 1
            end = ib[nxt] if ia[nxt] == end else ia[nxt]
            got.append(end)

    for s0 in range(len(ia)):
        if used[s0]:
            continue
        used[s0] = 1
        t = tile[s0]
        fwd = [ia[s0], ib[s0]] + walk(t, ib[s0])
        back = walk(t, ia[s0])
        out.append((t, [pos[i] for i in back[::-1] + fwd]))
    return out


# ---------------------------------------------------------------- Douglas-Peucker
def _dist(p, a, b):
    d = b - a
    L2 = float(d @ d)
    if L2 == 0.0:
        return np.hypot(*(p - a).T)
    return np.abs(d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])) / np.sqrt(L2)


def simplify(p, tol):
    """Douglas-Peucker. Returns the kept points and the largest distance of a dropped vertex from its span."""
    keep = simplify_mask(p, tol)
    return p[keep], span_dev(p, keep)


def span_dev(p, keep):
    """Independent re-check of every span: the largest distance of a dropped vertex from its span."""
    kept = np.nonzero(keep)[0]
    dev = 0.0
    for i, j in zip(kept[:-1], kept[1:]):
        if j > i + 1:
            dev = max(dev, float(_dist(p[i + 1:j], p[i], p[j]).max()))
    return dev


def simplify_mask(p, tol):
    """Douglas-Peucker keep mask."""
    m = len(p)
    keep = np.zeros(m, bool); keep[0] = keep[-1] = True
    if m < 3:
        keep[:] = True
        return keep
    stack = [(0, m - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = _dist(p[i + 1:j], p[i], p[j])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return keep


# ---------------------------------------------------------------- the run
def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def run(grid, sp=1.0, use_gpu=True, witness_on=True, tol=DP_TOL_M):
    """grid: south-up float64, NaN = no data. Returns (receipt, tiles {(tx, ty): {k: [np (n,2) metres]}})."""
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    R, C = grid.shape
    ne = 2 * R * C
    ntx = max(1, -(-(C - 1) // TILE_M))
    z = xp.asarray(grid, dtype=xp.float64)
    g = np.asarray(grid, np.float64)
    ks = levels_for(grid) if np.isfinite(grid).any() else []
    tot = dict(segments_e=0, segments_p=0, only_e=0, only_p=0, saddles_e=0, saddles_p=0, odd_cells=0,
               crossings=0, unmatched_ends=0, max_dxy=0.0)
    wit = dict(segments=0, only_w=0, only_e=0, saddles=0, unmatched_ends=0, max_dxy=0.0, photons=0)
    worst_levels, tiles, raw = [], {}, {}
    pts_in = pts_out = 0
    max_dev = 0.0
    t_pair = t_wit = t_host = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in ks:
            L = k * STEP_M
            a = time.perf_counter()
            seg, tl = cpair.pair_level(xp, z, L)
            seg = tuple(to_host(s) for s in seg)
            t_pair += time.perf_counter() - a
            for key in tot:
                tot[key] = max(tot[key], tl[key]) if key == "max_dxy" else tot[key] + tl[key]
            ph = tl["only_e"] + tl["only_p"] + tl["unmatched_ends"] + tl["odd_cells"] + int(tl["max_dxy"] > cpair.TOL_XY)
            if ph:
                worst_levels.append(dict(level_m=L, photons=ph, only_e=tl["only_e"], only_p=tl["only_p"]))
            if witness_on:
                a = time.perf_counter()
                w = cpair.witness_level(g, L, seg)
                for key in wit:
                    wit[key] = max(wit[key], w[key]) if key == "max_dxy" else wit[key] + w[key]
                t_wit += time.perf_counter() - a
            a = time.perf_counter()
            rr, cc = seg[0], seg[1]
            tile = (rr // TILE_M) * ntx + cc // TILE_M
            for t, pts in chain(tile, seg[2], seg[3], seg[4], seg[5], seg[6], seg[7], ne):
                p = np.asarray(pts, np.float64) * sp
                raw.setdefault((t % ntx, t // ntx), []).append((k, p, simplify_mask(p, tol)))
            t_host += time.perf_counter() - a
    a = time.perf_counter()
    topo = dict(found=0, left=0, touch=0, restored=0)
    for (tx, ty), got in raw.items():                  # topology: no two stored lines may touch or cross
        f, l_, u_, r_ = topo_mod.untangle([(p, keep) for _, p, keep in got], tx * TILE_M * sp, ty * TILE_M * sp)
        topo["found"] += f; topo["left"] += l_; topo["touch"] += u_; topo["restored"] += r_
        for k, p, keep in got:
            pts_in += len(p); pts_out += int(keep.sum()); max_dev = max(max_dev, span_dev(p, keep))
            tiles.setdefault((tx, ty), {}).setdefault(k, []).append(p[keep])
    t_host += time.perf_counter() - a
    photons = (tot["only_e"] + tot["only_p"] + tot["unmatched_ends"] + tot["odd_cells"] + int(tot["max_dxy"] > cpair.TOL_XY)
               + topo["left"])
    rec = dict(device=device, nodes=int(grid.size), levels=len(ks),
               level_range_m=[ks[0] * STEP_M, ks[-1] * STEP_M] if ks else None, step_m=STEP_M,
               electron=dict(method="marching squares, corner-case table, weighted in-cell crossing",
                             segments=tot["segments_e"], saddles=tot["saddles_e"]),
               positron=dict(method="row and column crossings found independently, joined by edge mask",
                             crossings=tot["crossings"], segments=tot["segments_p"], saddles=tot["saddles_p"]),
               photons=photons, only_electron=tot["only_e"], only_positron=tot["only_p"],
               odd_cells=tot["odd_cells"], unmatched_ends=tot["unmatched_ends"], max_dxy_grid=tot["max_dxy"],
               tol_xy_grid=cpair.TOL_XY, photon_levels=worst_levels[:12],
               simplify=dict(method="Douglas-Peucker, then dropped vertices put back until no stored lines cross",
                             tol_m=tol, tol_basis=TOL_BASIS, points_in=pts_in, points_out=pts_out,
                             max_dp_dev_m=round(max_dev, 6), within_tol=bool(max_dev <= tol + 1e-12),
                             crossings_found=topo["found"], vertices_restored=topo["restored"],
                             crossings=topo["left"], rounding_touches=topo["touch"]),
               timing_s=dict(pair=round(t_pair, 3), witness=round(t_wit, 3), chain_simplify=round(t_host, 3)))
    if witness_on:
        rec["witness"] = wit
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    return rec, tiles


# ---------------------------------------------------------------- tiles and index
def _dumps(obj):
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def encode_tile(levels, e0, n0, x0, y0, tol=DP_TOL_M):
    """levels {k: [(n,2) metres in site-local units]} -> ggc1 bytes, stats."""
    out, stats = [], {iv: dict(lines=0, length_m=0.0) for iv in INTERVALS}
    for k in sorted(levels):
        lines = []
        for q in levels[k]:
            cm = np.rint((q - (x0, y0)) * 100).astype(np.int64)
            lines.append(cm.ravel().tolist())
            ln = float(np.hypot(*np.diff(q, axis=0).T).sum()) if len(q) > 1 else 0.0
            for iv in INTERVALS:
                if in_interval(k, iv):
                    stats[iv]["lines"] += 1; stats[iv]["length_m"] += ln
        out.append(dict(z=k * STEP_M, lines=lines))
    doc = dict(format="ggc1", e0=int(e0), n0=int(n0), tile_m=TILE_M, unit_m=0.01, dp_tol_m=tol, levels=out)
    return _dumps(doc) + b"\n", stats


def write_tiles(out_dir, tiles, oe, on, site_name, sp=1.0, tol=DP_TOL_M):
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    entries = []
    for (tx, ty) in sorted(tiles, key=lambda t: (t[1], t[0])):
        levels = tiles[(tx, ty)]
        x0, y0 = tx * TILE_M * sp, ty * TILE_M * sp
        blob, st = encode_tile(levels, oe + x0, on + y0, x0, y0, tol)
        key = f"{tx}_{ty}"; rel = f"tiles/{key}.json"
        with open(os.path.join(out_dir, rel), "wb") as f:
            f.write(blob)
        ks = sorted(levels)
        entries.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(oe + x0),
                            n0=int(on + y0), bytes=len(blob), z_min=ks[0] * STEP_M, z_max=ks[-1] * STEP_M,
                            by_interval={str(iv): dict(lines=v["lines"], length_m=round(v["length_m"], 1))
                                         for iv, v in st.items()}))
    index = dict(format="ggc1", crs="EPSG:27700", site=dict(name=site_name, origin_e=int(oe), origin_n=int(on)),
                 tile_m=TILE_M, step_m=STEP_M, intervals_m=list(INTERVALS), index_every_m=5.0,
                 dp_tol_m=tol, dp_tol_basis=TOL_BASIS, topology="no two stored lines touch or cross", unit_m=0.01, heights="metres above Ordnance Datum Newlyn",
                 caveat=TERRAIN_ONLY, **ea_notice(), tiles=entries, generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True, help="folder with source.npy and source.json")
    ap.add_argument("--out", help="default SITE/contours")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-witness", action="store_true")
    a = ap.parse_args(argv)
    meta = json.load(open(os.path.join(a.site, "source.json"), encoding="utf-8"))
    grid = np.load(os.path.join(a.site, "source.npy"))
    if meta.get("rows", "south-to-north") != "south-to-north":
        raise SystemExit("source rows must run south to north")
    oe, on, sp = meta["origin_e_m"], meta["origin_n_m"], float(meta.get("spacing_m", 1))
    rec, tiles = run(grid, sp, use_gpu=not a.cpu, witness_on=not a.no_witness)
    out = a.out or os.path.join(a.site, "contours")
    index = write_tiles(out, tiles, oe, on, os.path.basename(os.path.normpath(a.site)), sp)
    here = os.path.dirname(os.path.abspath(__file__))
    rec.update(tiles=len(index["tiles"]), tile_bytes=sum(t["bytes"] for t in index["tiles"]),
               script_sha256={n: hashlib.sha256(open(os.path.join(here, n), "rb").read()).hexdigest()
                              for n in ("contour_tiles.py", "contour_pair.py", "contour_topo.py")},
               index_sha256=hashlib.sha256(open(os.path.join(out, INDEX), "rb").read()).hexdigest())
    with open(os.path.join(out, RECEIPT), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps({k: rec.get(k) for k in ("device", "levels", "level_range_m", "photons", "only_electron",
                                              "only_positron", "max_dxy_grid", "electron", "positron", "simplify",
                                              "witness", "tiles", "tile_bytes", "timing_s", "wall_s",
                                              "index_sha256")}, indent=1))
    ok = rec["photons"] == 0 and rec.get("witness", {}).get("photons", 0) == 0 and rec["simplify"]["within_tol"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
