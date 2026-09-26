"""Flow tiles from a south-up 1 m height grid: fill, route and accumulate as a pair on the GPU.

Where would water go, and where would it sit? Two routings, photons back (the electron/positron
pattern of pair_gpu.py and slope_tiles.py):

  fill       depressions filled two ways over the same fixed point. The result is the one
             priority-flood gives: every cell's water level W = Z where the ground is above its
             lowest neighbour's level m, else m (flat fill, for ponding) or m + EPS (drainable
             fill, for routing; EPS = 1e-5 m, so every interior cell has a lower neighbour).
             GPU: raster sweeps (S-N, N-S, W-E, E-W) from W = +inf down to the fixed point, run to
             convergence (Planchon-Darboux relaxation; same answer as priority-flood). CPU witness:
             a real priority-flood with a heap. The full grid is also checked against the fixed-
             point equations by a third, elementwise code path (fixpoint_violations must be 0).
  electron   D8 on the drainable fill: all of a cell's flow to its steepest-drop neighbour
             (drop / distance, diagonals sqrt 2 further; ties go to the first of N NE E SE S SW W NW).
  positron   D-infinity (Tarboton 1997) on the same fill: the steepest of 8 triangular facets,
             flow split between the facet's cardinal and diagonal neighbour by angle.
  accumulate each cell gives its own area (spacing^2) plus all it receives; on the GPU by
             topological peeling (Kahn: a cell sends once everything upstream has arrived), on the
             CPU witness by visiting cells from highest level to lowest. D8 also carries the longest
             upstream flow path in metres.
  photons    cells whose "channel" class (accumulation >= CHANNEL_M2 = 2,000 m^2) differs between
             D8 and D-inf. COUNTED and located, never suppressed. D-inf spreads flow, so photons
             are expected along the fringes of channels and on flats; the receipt says how many sit
             within 2 m of a D8 channel cell against the share expected by chance.
  ponding    depth = flat fill - ground; 8-connected wet cells grouped into depressions with
             volume, area, max depth and centroid. The biggest are listed in the receipt.
  witness    one 257 x 257 window (the tile with the most photons) run again end to end on the
             CPU in NumPy/heapq as a separate site, compared with the GPU run of the same window:
             fills, directions, both accumulations, channel classes, longest path, ponding.

Bounded and thermal-safe: fill rounds, peel levels and wall time are capped (MAX_*), the card
does a few seconds of work, and nothing loops without a convergence check.

.gfl v1 ("GGF1"), little-endian, 32-byte header then two u8 bodies:
    0 char[4] "GGF1" | 4 u16 version=1 | 6 u16 samples=257 | 8 u16 spacing_mm | 10 u16 flags=0
   12 i32 origin_e_m | 16 i32 origin_n_m (SW corner) | 20 u8 log_scale=10 | 21 u8 method=1 (D8)
   22 u16 channel_m2 (2000) | 24 u32 nodata_count | 28 u32 channel_count (D8 acc >= channel_m2)
   32 u8[257*257] accumulation class k = floor(10 * log10(area m^2)) clipped 0..254 (area >= 10^(k/10)),
      then u8[257*257] D8 direction 0..7 = N NE E SE S SW W NW, 8 = outlet (site edge), 255 = no data.
   Rows SOUTH to NORTH, each WEST to EAST. Tiles sit on the .ght grid (256 m, shared edges),
   indexed in flow-tiles.json with sha256s; the receipt lands beside it as flow_receipt.json.

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/flow_tiles.py --site E:/lidar-out/open-land-01
"""
import argparse, hashlib, json, math, os, sys, time
from datetime import datetime, timezone
import numpy as np
import flow_route
from flow_route import (  # noqa: F401  (re-exported: tests and callers use flow_tiles.*)
    cp, TILE_M, N, EPS, CHANNEL_M2, LOG_SCALE, NODATA, OUTLET, DR, DC, MAX_SECONDS, TOL_WIT, MAGIC, HEADER,
    INDEX, RECEIPT, masks, gpu_fill, gpu_route, gpu_accumulate, cpu_fill, cpu_route, cpu_accumulate, _nb,
    to_host, pipeline)


# ---------------------------------------------------------------- checks and readings (host NumPy)
def fixpoint_violations(grid, W, nod, fixed, eps):
    """Third code path: W must equal (Z if Z > m else m + eps) at every free cell, and Z on outlets."""
    free = ~(fixed | nod)
    Wm = np.where(nod, np.inf, W)
    m = np.min(np.stack([_nb(Wm, k) for k in range(8)]), 0)
    z = grid[1:-1, 1:-1]
    want = np.where(z > m, z, m + eps)
    bad = free[1:-1, 1:-1] & (W[1:-1, 1:-1] != want)
    return int(bad.sum()) + int((fixed & (W != grid)).sum())


def classes(acc, nod):
    k = np.floor(LOG_SCALE * np.log10(np.maximum(acc, 1e-300)))
    k = np.clip(k, 0, 254).astype(np.uint8)
    k[nod] = NODATA
    return k


def dilate(m, r):
    out = m.copy()
    for _ in range(r):
        o = out.copy()
        o[1:, :] |= out[:-1, :]; o[:-1, :] |= out[1:, :]
        o2 = o.copy(); o2[:, 1:] |= o[:, :-1]; o2[:, :-1] |= o[:, 1:]
        out = o2
    return out


def label8(mask):
    """8-connected components (GPU via cupyx when present, else a host flood)."""
    if cp is not None:
        import cupyx.scipy.ndimage as ndi
        lab, n = ndi.label(cp.asarray(mask), structure=cp.ones((3, 3), cp.int32))
        return to_host(lab), int(n)
    R, C = mask.shape; lab = np.zeros(mask.shape, np.int32); n = 0
    for s in zip(*np.nonzero(mask)):
        if lab[s]:
            continue
        n += 1; lab[s] = n; stack = [s]
        while stack:
            r, c = stack.pop()
            for dr, dc in zip(DR, DC):
                rr, cc = r + dr, c + dc
                if 0 <= rr < R and 0 <= cc < C and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n; stack.append((rr, cc))
    return lab, n


def ponding(grid, Wf, nod, sp, oe, on, keep=8):
    depth = np.where(nod, 0.0, Wf - grid)
    wet = depth > 0
    lab, n = label8(wet)
    area = sp * sp
    if n == 0:
        return dict(depressions=0, wet_cells=0, total_m3=0.0, max_depth_m=0.0, biggest=[]), depth
    flat = lab.ravel(); d = depth.ravel()
    vol = np.bincount(flat, weights=d, minlength=n + 1)[1:] * area
    cnt = np.bincount(flat, minlength=n + 1)[1:]
    rr, cc = np.divmod(np.arange(flat.size), grid.shape[1])
    sr = np.bincount(flat, weights=rr, minlength=n + 1)[1:]; sc = np.bincount(flat, weights=cc, minlength=n + 1)[1:]
    mx = np.zeros(n + 1); np.maximum.at(mx, flat[flat > 0], d[flat > 0]); mx = mx[1:]
    top = np.argsort(-vol, kind="stable")[:keep]
    big = [dict(volume_m3=round(float(vol[k]), 3), area_m2=round(float(cnt[k] * area), 1),
                max_depth_m=round(float(mx[k]), 3), level_m=round(float(Wf.ravel()[flat == k + 1][0]), 3),
                centroid_e=round(oe + (sc[k] / cnt[k]) * sp, 1), centroid_n=round(on + (sr[k] / cnt[k]) * sp, 1))
           for k in top]
    return dict(depressions=int(n), deeper_than_0p1m=int((mx >= 0.1).sum()), over_1m3=int((vol >= 1).sum()),
                wet_cells=int(wet.sum()), total_m3=round(float(vol.sum()), 3),
                max_depth_m=round(float(mx.max()), 4), biggest=big), depth


def longest_path(o, oe, on, sp):
    """End cell of the longest D8 path, then walk upstream along the edge that set its length."""
    lenE, R, C = o["lenE"], *o["grid"].shape
    L = lenE.ravel(); tgt = o["tgtE"][0::2]; stp = o["stepE"][0::2]
    end = int(np.argmax(L))
    cur, steps = end, 0
    while True:
        r, c = divmod(cur, C); nxt = -1
        for dr, dc in zip(DR, DC):
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C:
                u = rr * C + cc
                if tgt[u] == cur and L[u] + stp[u] == L[cur]:
                    nxt = u; break
        if nxt < 0:
            break
        cur = nxt; steps += 1
    re_, ce = divmod(end, C); rs, cs = divmod(cur, C)
    return dict(length_m=round(float(L[end]), 3), cells=steps + 1,
                source_e=oe + cs * sp, source_n=on + rs * sp, outlet_e=oe + ce * sp, outlet_n=on + re_ * sp,
                straight_m=round(math.hypot((ce - cs) * sp, (re_ - rs) * sp), 1))


def compare(o, oe, on, sp):
    """Photons: channel class disagreements between electron (D8) and positron (D-inf)."""
    nod = o["nod"]; valid = ~nod
    ce = (o["accE"] >= CHANNEL_M2) & valid; cpn = (o["accP"] >= CHANNEL_M2) & valid
    ph = ce ^ cpn
    nv = int(valid.sum()); nph = int(ph.sum())
    near = dilate(ce, 2) & valid
    rr, cc = np.nonzero(ph)
    nc16 = (ph.shape[1] + 15) // 16
    cnt = np.bincount((rr // 16) * nc16 + cc // 16, minlength=1)
    top = np.argsort(-cnt, kind="stable")[:8]
    hot = [((int(k) // nc16, int(k) % nc16), int(cnt[k])) for k in top if cnt[k] > 0]
    return dict(photons=nph, photon_share=round(nph / nv, 6) if nv else 0.0,
                share_of_channel_union=round(nph / max(int((ce | cpn).sum()), 1), 4),
                d8_only=int((ce & ~cpn).sum()), dinf_only=int((cpn & ~ce).sum()),
                within_2m_of_d8_channel=int((ph & near).sum()),
                share_near_channel=round(int((ph & near).sum()) / nph, 4) if nph else None,
                near_by_chance=round(int(near.sum()) / nv, 4) if nv else 0.0,
                hot_16m_cells=[dict(e=int(oe + c * 16 * sp), n=int(on + r * 16 * sp), photons=int(v)) for (r, c), v in hot]
                ), ph


def witness(grid, sp, deadline):
    """The same window end to end on the GPU and on the CPU (heap, NumPy, sorted sweep): compare."""
    g = pipeline(grid, sp, True, deadline) if cp is not None else None
    c = pipeline(grid, sp, False, deadline)
    rep = dict(nodes=int(grid.size), cpu_s=round(sum(c["times"].values()), 3))
    if g is None:
        rep["gpu"] = "absent"; return rep, c
    valid = ~c["nod"]
    dfill = float(np.abs(g["Wf"] - c["Wf"])[valid].max()); dfe = float(np.abs(g["We"] - c["We"])[valid].max())
    dir_mis = int((g["dir"] != c["dir"]).sum())
    dE = float(np.abs(g["accE"] - c["accE"])[valid].max()); dL = float(np.abs(g["lenE"] - c["lenE"])[valid].max())
    relP = float((np.abs(g["accP"] - c["accP"]) / np.maximum(c["accP"], 1e-300))[valid].max())
    clsE = int(((g["accE"] >= CHANNEL_M2) != (c["accE"] >= CHANNEL_M2))[valid].sum())
    # a D-inf class may flip only where accumulation sits within TOL_WIT (relative) of the threshold
    tieP = np.abs(c["accP"] - CHANNEL_M2) <= TOL_WIT * CHANNEL_M2
    flipP = ((g["accP"] >= CHANNEL_M2) != (c["accP"] >= CHANNEL_M2)) & valid
    pg, _ = ponding(grid, g["Wf"], g["nod"], sp, 0, 0); pc, _ = ponding(grid, c["Wf"], c["nod"], sp, 0, 0)
    rep.update(fill_flat_max_diff=dfill, fill_eps_max_diff=dfe, d8_dir_mismatch=dir_mis,
               d8_acc_max_diff=dE, d8_len_max_diff=dL, dinf_acc_max_rel_diff=relP,
               channel_mismatch_d8=clsE, channel_mismatch_dinf=int((flipP & ~tieP).sum()),
               dinf_threshold_ties=int((flipP & tieP).sum()),
               ponding_m3=[pg["total_m3"], pc["total_m3"]], depressions=[pg["depressions"], pc["depressions"]])
    rep["photons"] = int((dfill > TOL_WIT) + (dfe > TOL_WIT) + dir_mis + (dE > 1e-6) + (dL > 1e-6)
                         + (relP > TOL_WIT) + clsE + rep["channel_mismatch_dinf"]
                         + (abs(pg["total_m3"] - pc["total_m3"]) > 1e-3) + (pg["depressions"] != pc["depressions"]))
    return rep, c


def run(grid, oe=0, on=0, sp=1.0, use_gpu=True, witness_on=True, window=None, seconds=MAX_SECONDS):
    """grid: south-up float64 (NaN = no data). Returns (receipt, outputs dict)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return _run(grid, oe, on, sp, use_gpu, witness_on, window, seconds)


def _run(grid, oe, on, sp, use_gpu, witness_on, window, seconds):
    t0 = time.perf_counter(); deadline = t0 + seconds
    gpu = use_gpu and cp is not None
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if gpu else "cpu (numpy)"
    o = pipeline(grid, sp, gpu, deadline)
    nod, fixed, valid = o["nod"], o["fixed"], ~o["nod"]
    nv = int(valid.sum()); total = nv * sp * sp
    term = fixed & valid                                     # every drop ends on an outlet
    pond, depth = ponding(o["grid"], o["Wf"], nod, sp, oe, on)
    cmp_, ph = compare(o, oe, on, sp)
    lp = longest_path(o, oe, on, sp)
    iE = int(np.argmax(np.where(term, o["accE"], -1))); iP = int(np.argmax(np.where(term, o["accP"], -1)))
    C = grid.shape[1]
    rec = dict(
        device=device, nodes=int(grid.size), valid=nv, spacing_m=sp, eps_m=EPS, channel_m2=CHANNEL_M2,
        fill=dict(method_gpu="raster-sweep relaxation to the priority-flood fixed point" if gpu else "priority-flood",
                  rounds_flat_eps=o["fill_rounds"],
                  fixpoint_violations=fixpoint_violations(o["grid"], o["Wf"], nod, fixed, 0.0)
                  + fixpoint_violations(o["grid"], o["We"], nod, fixed, EPS),
                  raised_cells=int(((o["We"] > o["grid"]) & valid).sum()),
                  max_eps_lift_m=round(float((o["We"] - o["Wf"])[valid].max()), 6)),
        ponding=pond,
        electron=dict(method="d8", sinks=o["sinksE"], channel_cells=int(((o["accE"] >= CHANNEL_M2) & valid).sum()),
                      max_acc_m2=round(float(o["accE"][valid].max()), 1),
                      max_outlet=dict(e=oe + (iE % C) * sp, n=on + (iE // C) * sp),
                      mass_balance=round(float(o["accE"][term].sum()) / total, 12), longest_path=lp),
        positron=dict(method="dinf-tarboton", sinks=o["sinksP"],
                      channel_cells=int(((o["accP"] >= CHANNEL_M2) & valid).sum()),
                      max_acc_m2=round(float(o["accP"][valid].max()), 1),
                      max_outlet=dict(e=oe + (iP % C) * sp, n=on + (iP // C) * sp),
                      mass_balance=round(float(o["accP"][term].sum()) / total, 12)),
        peel_levels=o["levels"], times=o["times"], **cmp_)
    if witness_on:
        R = grid.shape[0]
        if window is None:
            if R <= N and grid.shape[1] <= N:
                window = (0, 0, R, grid.shape[1])
            else:                                            # the 256 m tile with the most photons
                ty, tx = max(((iy, ix) for iy in range((R - 1) // TILE_M) for ix in range((C - 1) // TILE_M)),
                             key=lambda t: int(ph[t[0] * TILE_M:t[0] * TILE_M + N, t[1] * TILE_M:t[1] * TILE_M + N].sum()))
                window = (ty * TILE_M, tx * TILE_M, N, N)
        r0, c0, h, w = window
        wr, _ = witness(o["grid"][r0:r0 + h, c0:c0 + w], sp, deadline)
        wr["window"] = dict(e0=oe + c0 * sp, n0=on + r0 * sp, rows=h, cols=w)
        rec["witness"] = wr
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    o.update(depth=depth, photon=ph)
    return rec, o


# ---------------------------------------------------------------- tiles
def encode(acls, d8, e0, n0, spacing_mm=1000, channel=None):
    if acls.shape != (N, N) or d8.shape != (N, N):
        raise ValueError(f"flow tile must be {N}x{N}")
    nod = int((d8 == NODATA).sum())
    kmin = math.floor(LOG_SCALE * math.log10(CHANNEL_M2))
    chan = int(channel) if channel is not None else int(((acls >= kmin) & (acls != NODATA)).sum())
    head = HEADER.pack(MAGIC, 1, N, spacing_mm, 0, int(e0), int(n0), LOG_SCALE, 1, int(CHANNEL_M2), nod, chan)
    return head + np.ascontiguousarray(acls, np.uint8).tobytes() + np.ascontiguousarray(d8, np.uint8).tobytes()


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError("short .gfl blob")
    magic, ver, n, sp, flags, e0, n0, scale, method, chm2, nod, chan = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != 1:
        raise ValueError(f"bad magic/version {magic!r} {ver}")
    if len(blob) != HEADER.size + 2 * n * n:
        raise ValueError(f".gfl size {len(blob)} != {HEADER.size + 2 * n * n}")
    body = np.frombuffer(blob, np.uint8, offset=HEADER.size)
    head = dict(samples=n, spacing_mm=sp, origin_e_m=e0, origin_n_m=n0, log_scale=scale, method=method,
                channel_m2=chm2, nodata_count=nod, channel_count=chan)
    return head, body[:n * n].reshape(n, n), body[n * n:].reshape(n, n)


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def write_tiles(out_dir, o, oe, on, site_name, spacing_mm=1000):
    acls = classes(o["accE"], o["nod"]); d8 = o["dir"]
    rows, cols = acls.shape
    if (rows - 1) % TILE_M or (cols - 1) % TILE_M:
        raise ValueError(f"grid {acls.shape} is not k*256+1 per side")
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    chan = (o["accE"] >= CHANNEL_M2) & ~o["nod"]; ph = o["photon"]; depth = o["depth"]
    entries = []
    for iy in range((rows - 1) // TILE_M):
        for ix in range((cols - 1) // TILE_M):
            r0, c0 = iy * TILE_M, ix * TILE_M
            sl = (slice(r0, r0 + N), slice(c0, c0 + N))
            blob = encode(acls[sl], d8[sl], oe + c0, on + r0, spacing_mm, chan[sl].sum())
            key = f"{ix}_{iy}"; rel = f"tiles/{key}.gfl"
            with open(os.path.join(out_dir, rel), "wb") as f:
                f.write(blob)
            entries.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(oe + c0),
                                n0=int(on + r0), bytes=len(blob), channel=int(chan[sl].sum()),
                                photons=int(ph[sl].sum()), max_log_class=int(acls[sl][acls[sl] != NODATA].max()),
                                pond_m3=round(float(depth[sl].sum()) * (spacing_mm / 1000) ** 2, 3)))
    index = dict(format="gfl1", crs="EPSG:27700", site=dict(name=site_name, origin_e=int(oe), origin_n=int(on)),
                 tile_m=TILE_M, spacing_m=spacing_mm / 1000, method="d8 on eps-filled surface",
                 accumulation=f"u8 k = floor({LOG_SCALE}*log10(area m2)), area >= 10^(k/{LOG_SCALE}); 255 no data",
                 direction="0..7 = N NE E SE S SW W NW, 8 outlet (site edge), 255 no data",
                 channel_m2=CHANNEL_M2, licence="OGL v3.0, see DATA-LICENCE.md",
                 attribution="© Environment Agency copyright and/or database right 2022. All rights reserved.",
                 tiles=entries, generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    dump_json(os.path.join(out_dir, INDEX), index)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True, help="folder with source.npy and source.json")
    ap.add_argument("--out", help="default SITE/flow")
    ap.add_argument("--seconds", type=float, default=MAX_SECONDS)
    a = ap.parse_args(argv)
    if cp is None:
        raise SystemExit("flow_tiles needs the GPU for a full site (the CPU path is the witness)")
    meta = json.load(open(os.path.join(a.site, "source.json")))
    grid = np.load(os.path.join(a.site, "source.npy"))
    if meta.get("rows", "south-to-north") != "south-to-north":
        raise SystemExit("source rows must run south to north")
    oe, on, sp = meta["origin_e_m"], meta["origin_n_m"], float(meta.get("spacing_m", 1))
    rec, o = run(grid, oe, on, sp, seconds=a.seconds)
    out = a.out or os.path.join(a.site, "flow")
    index = write_tiles(out, o, oe, on, os.path.basename(os.path.normpath(a.site)), int(round(sp * 1000)))
    rec.update(tiles=len(index["tiles"]), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               route_sha256=hashlib.sha256(open(flow_route.__file__, "rb").read()).hexdigest(),
               source_sha256=hashlib.sha256(open(os.path.join(a.site, "source.npy"), "rb").read()).hexdigest(),
               index_sha256=hashlib.sha256(open(os.path.join(out, INDEX), "rb").read()).hexdigest())
    dump_json(os.path.join(out, RECEIPT), rec)
    show = {k: rec[k] for k in ("device", "valid", "fill", "photons", "photon_share", "share_of_channel_union",
                                "share_near_channel", "near_by_chance", "peel_levels", "times", "wall_s")}
    show.update(longest=rec["electron"]["longest_path"], pond_total=rec["ponding"]["total_m3"],
                biggest=rec["ponding"]["biggest"][:3], mass=(rec["electron"]["mass_balance"], rec["positron"]["mass_balance"]),
                witness=rec.get("witness"), hot=rec["hot_16m_cells"][:4])
    print(json.dumps(show, indent=1))
    ok = rec.get("witness", {}).get("photons", 0) == 0 and rec["fill"]["fixpoint_violations"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
