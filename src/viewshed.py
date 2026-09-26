"""Visibility (viewshed) tiles from a south-up 1 m height grid, computed as a pair on the GPU.

Question answered: from which ground cells would a target of a given height (default 3 m, a panel top) be
seen by at least one person standing at an observer point (default eye height 1.7 m)? Observers are a list
of points, or points spaced along lines (roads, public footpaths) from JSON files.

  electron   R3: an exact sight line from the observer to every cell (one GPU thread per cell).
  positron   R2: a radial sweep, one ray to every border node of the grid; each cell is judged against
             the running horizon of the ray passing nearest it (one GPU thread per ray).
             Both from Franklin, W.R. and Ray, C.K. (1994) "Higher isn't necessarily better: visibility
             algorithms and experiments", Proc. 6th International Symposium on Spatial Data Handling,
             Edinburgh, pp. 751-770. R2 is the fast approximation of R3.
  photons    (observer, cell) pairs where R2 and R3 disagree. COUNTED and located, never suppressed.
             R2 judges a cell by a ray that passes up to half a cell from it, so photons are expected on
             the edges of visible areas; the receipt says what share sit on an edge against chance.
  witness    both methods again on the CPU in NumPy (viewshed_cpu.py, whole-array, a different code path)
             for a spread of observers; float32 in the same operation order with FMA off, so the maps
             must match to the bit.

Earth curvature and refraction: a point at horizontal distance d is lowered by (1 - k) d^2 / (2 R), with
R = 6 371 000 m (the mean Earth radius R1 of GRS80: Moritz, H. (2000) "Geodetic Reference System 1980",
Journal of Geodesy 74, 128-133) and k = 0.13, the conventional coefficient of terrestrial refraction
(Torge, W. and Mueller, J. (2012) Geodesy, 4th edn, de Gruyter, section 5.1). At 1 km this is 6.8 cm;
at 3 km, 61 cm.

The published tiles are the electron (exact) counts. Terrain is the bare-earth DTM subsampled to the
analysis spacing (default 2 m, every second node): hedges, trees and buildings are NOT included.

.gvs v1 ("GGV1"), little-endian, 32-byte header then one u8 body:
    0 char[4] "GGV1" | 4 u16 version=1 | 6 u16 samples | 8 u16 spacing_mm | 10 u16 flags (bit 0 curvature,
      bit 1 canopy) | 12 i32 origin_e_m | 16 i32 origin_n_m (SW corner) | 20 u16 eye_mm | 22 u16 target_mm
   24 u32 visible_count | 28 u32 nodata_count | 32 u8[samples*samples]: 0 hidden from every observer,
      1..254 number of observers who see it (254 = 254 or more), 255 no data; rows SOUTH to NORTH.

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/viewshed.py --site E:/lidar-out/open-land-01
"""
import argparse, hashlib, json, os, struct, sys, time
from datetime import datetime, timezone
import numpy as np
import viewshed_cpu as vc

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the same arithmetic runs in NumPy and says so
    cp = None

R_EARTH = 6371000.0
K_REFRACTION = 0.13
EYE_M, TARGET_M, CELL_M, OBS_SPACING_M, TILE_M = 1.7, 3.0, 2, 50.0, 256
NODATA, CAP = 255, 254
MAGIC = b"GGV1"
HEADER = struct.Struct("<4sHHHHiiHHII")
assert HEADER.size == 32
INDEX, RECEIPT = "visibility-tiles.json", "visibility_receipt.json"
CAVEAT = "computed from terrain only; hedges and trees not included"

KERNELS = r"""
__device__ __forceinline__ void fdiv(int a, int n, int* fl, int* rem) {
  int q = a / n, r = a - q * n; if (r < 0) { q -= 1; r += n; } *fl = q; *rem = r; }
// height where the line (dr, dc) from (ro, co) crosses major step k of n; base node and remainder out
__device__ __forceinline__ float sample(const float* z, int cols, int ro, int co, int dr, int dc, int k, int n,
                                        int* rr, int* cc, int* rem) {
  int fl; float z0, z1, f;
  if (abs(dc) >= abs(dr)) { fdiv(dr * k, n, &fl, rem); *rr = ro + fl; *cc = co + (dc > 0 ? k : -k);
    z0 = z[*rr * cols + *cc]; if (*rem == 0) return z0; z1 = z[(*rr + 1) * cols + *cc]; }
  else { fdiv(dc * k, n, &fl, rem); *rr = ro + (dr > 0 ? k : -k); *cc = co + fl;
    z0 = z[*rr * cols + *cc]; if (*rem == 0) return z0; z1 = z[*rr * cols + *cc + 1]; }
  f = (float)*rem / (float)n;
  return z0 + f * (z1 - z0);
}
extern "C" __global__ void r3(const float* z, int rows, int cols, float sp, int ro, int co, float eye,
                              float tgt, float curv, unsigned char* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x; if (idx >= rows * cols) return;
  int r = idx / cols, c = idx - r * cols; float zt = z[idx];
  if (isnan(zt)) { out[idx] = 255; return; }
  int dr = r - ro, dc = c - co, n = max(abs(dr), abs(dc));
  if (n == 0) { out[idx] = 1; return; }
  float zo = z[ro * cols + co] + eye, D = sp * sqrtf((float)(dr * dr + dc * dc)), smax = __int_as_float(0xff800000);
  int rr, cc, rem;
  for (int k = 1; k < n; k++) {
    float h = sample(z, cols, ro, co, dr, dc, k, n, &rr, &cc, &rem);
    float d = D * (float)k / (float)n;
    smax = fmaxf(smax, (h - curv * d * d - zo) / d);
  }
  float st = (zt + tgt - curv * D * D - zo) / D;
  out[idx] = st >= smax ? 1 : 0;
}
extern "C" __global__ void r2(const float* z, int rows, int cols, float sp, int ro, int co, float eye,
                              float tgt, float curv, const int* per, int nper, unsigned char* out) {
  int t = blockDim.x * blockIdx.x + threadIdx.x; if (t >= nper) return;
  int dr = per[2 * t] - ro, dc = per[2 * t + 1] - co, n = max(abs(dr), abs(dc));
  if (n == 0) return;
  float zo = z[ro * cols + co] + eye, DP = sp * sqrtf((float)(dr * dr + dc * dc)), smax = __int_as_float(0xff800000);
  int rr, cc, rem;
  for (int k = 1; k <= n; k++) {
    float h = sample(z, cols, ro, co, dr, dc, k, n, &rr, &cc, &rem);
    int up = 2 * rem >= n ? 1 : 0, tr = rr, tc = cc;
    if (abs(dc) >= abs(dr)) tr += up; else tc += up;
    int er = tr - ro, ec = tc - co;
    float Dc = sp * sqrtf((float)(er * er + ec * ec));
    float st = (z[tr * cols + tc] + tgt - curv * Dc * Dc - zo) / Dc;
    if (st >= smax) out[tr * cols + tc] = 1;
    float d = DP * (float)k / (float)n;
    smax = fmaxf(smax, (h - curv * d * d - zo) / d);
  }
}
"""
_mods = {}


def kernel(name):
    if name not in _mods:
        _mods[name] = cp.RawKernel(KERNELS, name, options=("--fmad=false",))
    return _mods[name]


def curvature_coeff(k=K_REFRACTION, R=R_EARTH):
    """c in drop = c * d^2."""
    return (1.0 - k) / (2.0 * R)


# ---------------------------------------------------------------- observers
def along(pts, spacing):
    """Points every `spacing` metres along a polyline, both ends included."""
    pts = np.asarray(pts, float)
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(b - a)))
        s = spacing - carry
        while s <= L + 1e-9:
            out.append(a + (b - a) * (s / L)); s += spacing
        carry = (carry + L) % spacing
    out.append(pts[-1])
    return out


def load_observers(paths, spacing=OBS_SPACING_M, points=()):
    """Observer points (easting, northing) from JSON files and loose points.

    Accepts: {"points": [[e, n], ...]}; {"lines": [{"pts": [[e, n], ...]}, ...]} (the ways files);
    GeoJSON FeatureCollection of Point / MultiPoint / LineString / MultiLineString in EPSG:27700."""
    out = [tuple(map(float, p)) for p in points]
    for path in paths:
        j = json.load(open(path, encoding="utf-8"))
        out += [tuple(map(float, p[:2])) for p in j.get("points", [])]
        for ln in j.get("lines", []):
            out += [tuple(p) for p in along(ln["pts"], spacing)]
        for ft in j.get("features", []):
            g = ft.get("geometry") or {}
            t, cs = g.get("type"), g.get("coordinates")
            if t == "Point":
                out.append(tuple(map(float, cs[:2])))
            elif t == "MultiPoint":
                out += [tuple(map(float, p[:2])) for p in cs]
            elif t in ("LineString", "MultiLineString"):
                for ln in ([cs] if t == "LineString" else cs):
                    out += [tuple(p) for p in along([p[:2] for p in ln], spacing)]
    return out


def snap(obs, oe, on, cell, rows, cols):
    """Nearest analysis node per observer, duplicates merged. Returns (nodes, dropped outside the grid)."""
    nodes, seen, dropped = [], set(), 0
    for e, n in obs:
        r, c = int(round((n - on) / cell)), int(round((e - oe) / cell))
        if not (0 <= r < rows and 0 <= c < cols):
            dropped += 1; continue
        if (r, c) not in seen:
            seen.add((r, c)); nodes.append((r, c))
    return nodes, dropped


# ---------------------------------------------------------------- one observer, both methods
def pair_one(xp, zd, sp, ro, co, eye, tgt, curv, per_d):
    rows, cols = zd.shape
    if xp is np:
        return vc.r3(zd, sp, ro, co, eye, tgt, curv), vc.r2(zd, sp, ro, co, eye, tgt, curv, per_d)
    f = np.float32
    e = cp.empty((rows, cols), cp.uint8)
    kernel("r3")(((rows * cols + 255) // 256,), (256,),
                 (zd, np.int32(rows), np.int32(cols), f(sp), np.int32(ro), np.int32(co), f(eye), f(tgt), f(curv), e))
    p = cp.zeros((rows, cols), cp.uint8)
    p[ro, co] = 1
    npr = per_d.shape[0]
    kernel("r2")(((npr + 127) // 128,), (128,),
                 (zd, np.int32(rows), np.int32(cols), f(sp), np.int32(ro), np.int32(co), f(eye), f(tgt), f(curv),
                  per_d, np.int32(npr), p))
    p[cp.isnan(zd)] = NODATA
    return e, p


def edges(xp, vis):
    """Cells whose electron verdict differs from a 4-neighbour: the edge of a visible area."""
    v = vis == 1
    e = xp.zeros_like(v)
    e[1:, :] |= v[1:, :] != v[:-1, :]; e[:-1, :] |= v[1:, :] != v[:-1, :]
    e[:, 1:] |= v[:, 1:] != v[:, :-1]; e[:, :-1] |= v[:, 1:] != v[:, :-1]
    return e & (vis != NODATA)


def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def run(grid, nodes, sp=1.0, eye=EYE_M, tgt=TARGET_M, curvature=True, k=K_REFRACTION, use_gpu=True,
        witness_n=3, oe=0, on=0):
    """grid: south-up heights at the analysis spacing sp (NaN = no data); nodes: [(row, col)] observers.
    Returns (receipt, electron counts u8, positron counts u8)."""
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    g = np.asarray(grid, np.float64)
    base = float(np.nanmin(g)) if np.isfinite(g).any() else 0.0
    z32 = (g - base).astype(np.float32)                   # heights above the lowest node: finer float32 steps
    curv = curvature_coeff(k) if curvature else 0.0
    zd = xp.asarray(z32)
    per = vc.perimeter(*z32.shape)
    per_d = xp.asarray(per)
    ce = xp.zeros(z32.shape, xp.int32); cpn = xp.zeros(z32.shape, xp.int32); heat = xp.zeros(z32.shape, xp.int32)
    nph = on_edge = edge_cells = 0
    valid = int(np.isfinite(g).sum())
    wit_idx = sorted(set(np.linspace(0, len(nodes) - 1, min(witness_n, len(nodes))).round().astype(int).tolist())) if nodes else []
    wit = dict(observers=len(wit_idx), r3_mismatch=0, r2_mismatch=0) if (witness_n and xp is not np) else None
    per_obs = []
    for i, (ro, co) in enumerate(nodes):
        e, p = pair_one(xp, zd, sp, ro, co, eye, tgt, curv, per_d)
        ce += (e == 1); cpn += (p == 1)
        dis = (e != p) & (e != NODATA)
        heat += dis
        ed = edges(xp, e)
        n_dis = int(dis.sum()); nph += n_dis
        on_edge += int((dis & ed).sum()); edge_cells += int(ed.sum())
        per_obs.append(int((e == 1).sum()))
        if wit is not None and i in wit_idx:
            we, wp = vc.r3(z32, sp, ro, co, eye, tgt, curv), vc.r2(z32, sp, ro, co, eye, tgt, curv, per)
            wit["r3_mismatch"] += int((we != to_host(e)).sum()); wit["r2_mismatch"] += int((wp != to_host(p)).sum())
    ce, cpn, heat = to_host(ce), to_host(cpn), to_host(heat)
    nod = ~np.isfinite(g)
    out_e = np.minimum(ce, CAP).astype(np.uint8); out_e[nod] = NODATA
    out_p = np.minimum(cpn, CAP).astype(np.uint8); out_p[nod] = NODATA
    any_e, any_p = (out_e >= 1) & ~nod, (out_p >= 1) & ~nod
    pairs = len(nodes) * valid
    hot = np.argsort(heat.ravel())[::-1][:8]
    maxd = sp * float(np.hypot(*z32.shape))
    rec = dict(
        device=device, grid=dict(rows=int(g.shape[0]), cols=int(g.shape[1]), spacing_m=sp, valid=valid),
        observers=len(nodes), eye_m=eye, target_m=tgt,
        curvature=dict(on=bool(curvature), R_m=R_EARTH, k=k, coeff_per_m=curv,
                       drop_1km_m=round(curv * 1e6, 4), drop_at_grid_diagonal_m=round(curv * maxd * maxd, 4)),
        electron=dict(method="R3 exact sight line", visible_any=int(any_e.sum()),
                      share_any=round(float(any_e.sum()) / valid, 6) if valid else 0.0,
                      mean_share_per_observer=round(float(np.mean(per_obs)) / valid, 6) if per_obs and valid else 0.0),
        positron=dict(method="R2 radial sweep", visible_any=int(any_p.sum()),
                      share_any=round(float(any_p.sum()) / valid, 6) if valid else 0.0),
        photons=nph, observer_cell_pairs=pairs, photon_share=round(nph / pairs, 8) if pairs else 0.0,
        union_photons=int((any_e != any_p).sum()),
        edges=dict(photons_on_edge=on_edge, share_on_edge=round(on_edge / nph, 4) if nph else None,
                   by_chance=round(edge_cells / pairs, 4) if pairs else None),
        hot_cells=[dict(e=int(oe + (h % g.shape[1]) * sp), n=int(on + (h // g.shape[1]) * sp), photons=int(heat.ravel()[h]))
                   for h in hot if heat.ravel()[h] > 0])
    if wit is not None:
        wit["photons"] = wit["r3_mismatch"] + wit["r2_mismatch"]
        rec["witness"] = wit
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    return rec, out_e, out_p


# ---------------------------------------------------------------- tiles
def encode(t, e0, n0, spacing_mm, eye, tgt, flags):
    s = t.shape[0]
    if t.shape != (s, s):
        raise ValueError("visibility tile must be square")
    vis = int(((t >= 1) & (t != NODATA)).sum()); nod = int((t == NODATA).sum())
    head = HEADER.pack(MAGIC, 1, s, spacing_mm, flags, int(e0), int(n0), int(round(eye * 1000)), int(round(tgt * 1000)), vis, nod)
    return head + np.ascontiguousarray(t, np.uint8).tobytes()


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError("short .gvs blob")
    magic, ver, s, sp, flags, e0, n0, eye, tgt, vis, nod = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != 1:
        raise ValueError(f"bad magic/version {magic!r} {ver}")
    if len(blob) != HEADER.size + s * s:
        raise ValueError(f".gvs size {len(blob)} != {HEADER.size + s * s}")
    head = dict(samples=s, spacing_mm=sp, flags=flags, origin_e_m=e0, origin_n_m=n0, eye_mm=eye, target_mm=tgt,
                visible_count=vis, nodata_count=nod)
    return head, np.frombuffer(blob, np.uint8, offset=HEADER.size).reshape(s, s)


def write_tiles(out_dir, counts, oe, on, cell, site_name, eye, tgt, observers, curvature=True, canopy=False, extra=None):
    step = int(round(TILE_M / cell))
    rows, cols = counts.shape
    if TILE_M % cell or (rows - 1) % step or (cols - 1) % step:
        raise ValueError(f"grid {counts.shape} at {cell} m is not k*{step}+1 per side")
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    flags = (1 if curvature else 0) | (2 if canopy else 0)
    entries = []
    for iy in range((rows - 1) // step):
        for ix in range((cols - 1) // step):
            r0, c0 = iy * step, ix * step
            t = counts[r0:r0 + step + 1, c0:c0 + step + 1]
            blob = encode(t, oe + c0 * cell, on + r0 * cell, int(round(cell * 1000)), eye, tgt, flags)
            key = f"{ix}_{iy}"; rel = f"tiles/{key}.gvs"
            with open(os.path.join(out_dir, rel), "wb") as f:
                f.write(blob)
            entries.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(oe + c0 * cell),
                                n0=int(on + r0 * cell), bytes=len(blob),
                                visible=int(((t >= 1) & (t != NODATA)).sum())))
    index = dict(format="gvs1", crs="EPSG:27700", site=dict(name=site_name, origin_e=int(oe), origin_n=int(on)),
                 tile_m=TILE_M, spacing_m=cell, eye_m=eye, target_m=tgt, method="R3 exact sight line (Franklin and Ray 1994)",
                 curvature=dict(on=bool(curvature), R_m=R_EARTH, k=K_REFRACTION), canopy=bool(canopy),
                 caveat=None if canopy else CAVEAT,
                 values="0 hidden from every observer; 1..254 observers who see it (254 = 254 or more); 255 no data",
                 observers=[[round(e, 1), round(n, 1)] for e, n in observers], tiles=entries,
                 generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **(extra or {}))
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True, help="folder with source.npy and source.json")
    ap.add_argument("--observers", action="append", help="JSON of points or lines (repeatable); default SITE/roads.json")
    ap.add_argument("--point", action="append", default=[], help="E,N observer (repeatable)")
    ap.add_argument("--spacing", type=float, default=OBS_SPACING_M, help="metres between observers along lines")
    ap.add_argument("--cell", type=int, default=CELL_M, help="analysis spacing in whole metres")
    ap.add_argument("--eye", type=float, default=EYE_M)
    ap.add_argument("--target", type=float, default=TARGET_M)
    ap.add_argument("--no-curvature", action="store_true")
    ap.add_argument("--witness", type=int, default=3, help="observers re-run on the CPU")
    ap.add_argument("--out", help="default SITE/visibility")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    meta = json.load(open(os.path.join(a.site, "source.json")))
    if meta.get("rows", "south-to-north") != "south-to-north":
        raise SystemExit("source rows must run south to north")
    oe, on, sp0 = meta["origin_e_m"], meta["origin_n_m"], float(meta.get("spacing_m", 1))
    stride = int(round(a.cell / sp0))
    grid = np.load(os.path.join(a.site, "source.npy"))[::stride, ::stride]
    paths = a.observers or ([] if a.point else [os.path.join(a.site, "roads.json")])
    obs = load_observers(paths, a.spacing, [tuple(map(float, p.split(","))) for p in a.point])
    nodes, dropped = snap(obs, oe, on, a.cell, *grid.shape)
    if not nodes:
        raise SystemExit("no observer falls inside the grid")
    rec, ce, _ = run(grid, nodes, float(a.cell), a.eye, a.target, not a.no_curvature, use_gpu=not a.cpu,
                     witness_n=a.witness, oe=oe, on=on)
    snapped = [(oe + c * a.cell, on + r * a.cell) for r, c in nodes]
    out = a.out or os.path.join(a.site, "visibility")
    index = write_tiles(out, ce, oe, on, a.cell, os.path.basename(os.path.normpath(a.site)), a.eye, a.target, snapped,
                        not a.no_curvature, extra=dict(observer_sources=[os.path.basename(p) for p in paths],
                                                       observer_spacing_m=a.spacing))
    rec.update(observer_points_in=len(obs), observers_outside_grid=dropped, tiles=len(index["tiles"]),
               script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               cpu_sha256=hashlib.sha256(open(vc.__file__, "rb").read()).hexdigest(),
               index_sha256=hashlib.sha256(open(os.path.join(out, INDEX), "rb").read()).hexdigest())
    with open(os.path.join(out, RECEIPT), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in rec if k != "hot_cells"}, indent=1))
    return 0 if rec.get("witness", {}).get("photons", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
