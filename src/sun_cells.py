"""Sun on every 4 m cell: monthly direct-beam hours and terrain-shaded direct irradiation, a pair on the GPU.

Inputs per site: horizon/horizon-tiles.json (+ .ghz tiles, 32 azimuths per 4 m cell, horizon_tiles.py) and
sun/sun-climate.json + sun/tmy-hourly.bin (sun_data.py, PVGIS TMY with cloud from SARAH-3 satellite data).
A cell is in terrain shadow when the sun's elevation is below its horizon in the sun's grid azimuth
(Dozier and Frew 1990). Bare earth: hedges, trees and buildings cast no shadow here. Irradiation is on the
horizontal plane; refraction is ignored; the sun path is for a non-leap reference year 2023, UTC.

Four quantities per cell and month:
  q0 clear-sky beam hours      hours with the sun above the terrain horizon
  q1 clear-sky direct kWh/m2   Meinel and Meinel (1976) clear-sky DNI with the Laue (1970) altitude term,
                               air mass Kasten and Young (1989), times sin(elevation), where unshaded
  q2 sunshine hours            TMY hours with Gb(n) >= 120 W/m2 (WMO) and the sun above the terrain horizon
  q3 TMY direct kWh/m2         TMY Gb(n) times sin(elevation), where unshaded (cloud-weighted)

Two channels, photons back:
  electron  hour by hour: one sun position per TMY hour at HH:00 + PVGIS offset (NOAA general solar position,
            Spencer 1971 series), horizon linear between the 32 azimuths at the sun's own azimuth, a CUDA kernel,
            float64, no fused multiply-add, one thread per cell.
  positron  a sky-map integral: ten 6-minute sun positions per hour centred on the same instant (Michalsky 1988,
            Astronomical Almanac algorithm), binned by month and 0.5 degree of grid azimuth; each bin holds its
            elevations sorted with suffix sums, so a cell reads its unshaded total by one binary search per bin
            against its horizon at the bin centre (CuPy array operations).
  photons   (cell, month, quantity) where the shaded share differs by more than 5 points between the channels.
            Counted and located, never suppressed.
  witness   a seeded sample of cells again on the CPU: the electron as NumPy over hours with the horizon as a
            lerp, the positron as a direct sum over every sub-hourly sample. Both must agree to 1e-9.

Output SITE/sun/cells/: tiles/<ix>_<iy>.gsc (256 m, 65 x 65 cells, edges shared), sun-cells.json index with
sha256s and the open-sky monthly base values, sun_cells_receipt.json. LF throughout.

.gsc v1, little-endian, 32-byte header:
    0 char[4] "GSC1" | 4 u16 version=1 | 6 u16 samples=65 | 8 u16 spacing_mm=4000 | 10 u16 months=12
   12 i32 origin_e_m | 16 i32 origin_n_m (SW corner) | 20 u16 quantities=4 | 22 u16 scale=250
   24 u32 nodata_cells | 28 u32 reserved=0
   32 u8[65*65*12*4] rows SOUTH to NORTH, each WEST to EAST; per cell 12 months x 4 quantities;
      value / 250 = share of the open-sky monthly base (in the index); 255 = horizon unknown where the sun goes.

    python src/sun_cells.py --site $LIDAR_OUT/open-land-01
"""
import argparse, hashlib, json, os, struct, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horizon_tiles as hz  # noqa: E402
import sun_data as sd  # noqa: E402

cp = hz.cp
NAZ = hz.NAZ
NQ = 4
SCALE = 250
NODATA_U8 = 255
TOL_SHARE = 0.05
TOL_WIT = 1e-9
MAGIC = b"GSC1"
HEADER = struct.Struct("<4sHHHHiiHHII")
assert HEADER.size == 32
QUANTITIES = ["clear_beam_hours", "clear_direct_kwh_m2", "sunshine_hours", "tmy_direct_kwh_m2"]


# ---------------------------------------------------------------- sun position: sun_geom.py
from sun_geom import (noaa, michalsky, clear_dni, hour_table,  # noqa: E402,F401
                      electron_samples, positron_samples, SUB, BIN_DEG, NBIN, REF_YEAR, MONTH_START)


# ---------------------------------------------------------------- electron
KERNEL = r"""
extern "C" __global__ void sunpath(const double* hz, const int ncell, const double* f, const double* el,
    const int* mon, const double* w, const int nh, double* out) {
  int c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= ncell) return;
  double acc[48];
  for (int q = 0; q < 48; q++) acc[q] = 0.0;
  const double* h = hz + (long long)c * 32;
  for (int i = 0; i < nh; i++) {
    int k = (int)floor(f[i]); double a = f[i] - k; k = k % 32;
    double hh = h[k] * (1.0 - a) + h[(k + 1) % 32] * a;
    if (el[i] < hh) continue;                      /* NaN horizon: never shaded */
    int m = mon[i] * 4;
    acc[m] += w[i * 4]; acc[m + 1] += w[i * 4 + 1]; acc[m + 2] += w[i * 4 + 2]; acc[m + 3] += w[i * 4 + 3];
  }
  for (int q = 0; q < 48; q++) out[(long long)c * 48 + q] = acc[q];
}
"""
_kernel = None


def electron_gpu(hzc, S, chunk=65536, guard=None):
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(KERNEL, "sunpath", options=("--fmad=false",))
    f, el, mon, w = (cp.asarray(S[k]) for k in ("f", "el", "month", "w"))
    out = np.empty((len(hzc), 12, NQ))
    for a in range(0, len(hzc), chunk):
        if guard:
            guard()
        h = cp.asarray(np.ascontiguousarray(hzc[a:a + chunk]), cp.float64)
        o = cp.empty((len(h), 48), cp.float64)
        _kernel(((len(h) + 127) // 128,), (128,), (h, np.int32(len(h)), f, el, mon, w.ravel(), np.int32(len(S["el"])), o))
        out[a:a + chunk] = cp.asnumpy(o).reshape(-1, 12, NQ)
    return out


def electron_cpu(hzc, S):
    """NumPy over hours, one cell at a time; the horizon as a lerp h0 + (h1 - h0) a."""
    k = np.floor(S["f"]).astype(np.int64); a = S["f"] - k; k %= NAZ; k1 = (k + 1) % NAZ
    out = np.zeros((len(hzc), 12, NQ))
    for c in range(len(hzc)):
        h0, h1 = hzc[c, k], hzc[c, k1]
        hh = h0 + (h1 - h0) * a
        lit = ~(S["el"] < hh)
        for m in range(12):
            s = lit & (S["month"] == m)
            out[c, m] = S["w"][s].sum(0)
    return out


# ---------------------------------------------------------------- positron
def sky_map(P):
    """Per (month, azimuth bin): elevations sorted ascending and suffix sums of the weights (n + 1 rows)."""
    key = P["month"] * NBIN + P["bin"]
    order = np.lexsort((P["el"], key))
    key, el, w = key[order], P["el"][order], P["w"][order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    out = {}
    for s, e in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(key)]])):
        suf = np.zeros((e - s + 1, NQ)); suf[:-1] = np.cumsum(w[s:e][::-1], 0)[::-1]
        out[int(key[s])] = (el[s:e], suf)
    return out


def horizon_at(xp, hzc, az_deg):
    f = az_deg / (360.0 / NAZ); k = int(np.floor(f)) % NAZ; a = f - np.floor(f)
    h = hzc[:, k] * (1 - a) + hzc[:, (k + 1) % NAZ] * a
    return xp.where(xp.isnan(h), -xp.inf, h)


def positron(xp, hzc, sky, guard=None):
    hzc = xp.asarray(hzc)
    out = xp.zeros((hzc.shape[0], 12, NQ))
    by_bin = {}
    for key, v in sky.items():
        by_bin.setdefault(key % NBIN, []).append((key // NBIN, v))
    for i, (b, items) in enumerate(sorted(by_bin.items())):
        if guard and i % 64 == 0:
            guard()
        h = horizon_at(xp, hzc, (b + 0.5) * BIN_DEG)
        for m, (el, suf) in items:
            idx = xp.searchsorted(xp.asarray(el), h, side="left")   # first sample with el >= h is lit
            out[:, m, :] += xp.asarray(suf)[idx]
    return out


def positron_direct(hzc, P):
    """Witness: every sub-hourly sample summed straight, horizon at its bin centre."""
    cen = (P["bin"] + 0.5) * BIN_DEG
    f = cen / (360.0 / NAZ); k = np.floor(f).astype(np.int64); a = f - k; k %= NAZ
    out = np.zeros((len(hzc), 12, NQ))
    for c in range(len(hzc)):
        hh = hzc[c, k] * (1 - a) + hzc[c, (k + 1) % NAZ] * a
        lit = ~(P["el"] < hh)                               # NaN never shades
        np.add.at(out[c], P["month"][lit], P["w"][lit])
    return out


# ---------------------------------------------------------------- tiles
def load_horizon(hdir):
    idx = json.load(open(os.path.join(hdir, hz.INDEX), encoding="utf-8"))
    step = hz.TILE_M // hz.CELL_M
    nx = 1 + max(int(t["key"].split("_")[0]) for t in idx["tiles"]); ny = 1 + max(int(t["key"].split("_")[1]) for t in idx["tiles"])
    g = np.full((ny * step + 1, nx * step + 1, NAZ), np.nan)
    for t in idx["tiles"]:
        blob = open(os.path.join(hdir, t["file"]), "rb").read()
        if hashlib.sha256(blob).hexdigest() != t["sha256"]:
            raise ValueError(f"horizon tile {t['file']} sha256 mismatch")
        _, q = hz.decode(blob)
        ix, iy = (int(v) for v in t["key"].split("_"))
        g[iy * step:iy * step + hz.NS, ix * step:ix * step + hz.NS] = np.where(q == hz.NODATA, np.nan, q / hz.UNIT)
    return g, idx


def encode(t, e0, n0):
    if t.shape != (hz.NS, hz.NS, 12, NQ):
        raise ValueError("sun cell tile must be 65x65x12x4")
    nod = int((t[..., 0, 0] == NODATA_U8).sum())
    head = HEADER.pack(MAGIC, 1, hz.NS, hz.CELL_M * 1000, 12, int(e0), int(n0), NQ, SCALE, nod, 0)
    return head + np.ascontiguousarray(t, np.uint8).tobytes()


def decode(blob):
    magic, ver, ns, sp, months, e0, n0, nq, scale, nod, _ = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != 1 or len(blob) != HEADER.size + ns * ns * months * nq:
        raise ValueError("bad .gsc blob")
    return (dict(samples=ns, spacing_mm=sp, months=months, origin_e_m=e0, origin_n_m=n0, quantities=nq,
                 scale=scale, nodata_cells=nod),
            np.frombuffer(blob, np.uint8, offset=HEADER.size).reshape(ns, ns, months, nq))


def quantise(val, base, nodata):
    share = np.where(base > 0, val / np.where(base > 0, base, 1), 1.0)
    q = np.clip(np.round(share * SCALE), 0, SCALE).astype(np.uint8)
    q[nodata] = NODATA_U8
    return q


def write_tiles(out, q, oe, on, meta):
    step = hz.TILE_M // hz.CELL_M
    rows, cols = q.shape[:2]
    os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
    entries = []
    for iy in range((rows - 1) // step):
        for ix in range((cols - 1) // step):
            t = q[iy * step:iy * step + hz.NS, ix * step:ix * step + hz.NS]
            e0, n0 = oe + ix * hz.TILE_M, on + iy * hz.TILE_M
            blob = encode(t, e0, n0); rel = f"tiles/{ix}_{iy}.gsc"
            with open(os.path.join(out, rel), "wb") as f:
                f.write(blob)
            v = t[t[..., 0, 0] != NODATA_U8]
            entries.append(dict(key=f"{ix}_{iy}", file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(e0),
                                n0=int(n0), bytes=len(blob), nodata_cells=int((t[..., 0, 0] == NODATA_U8).sum()),
                                min_share_annual_tmy_kwh=round(float(v[..., 3].min()) / SCALE, 3) if v.size else None))
    index = dict(format="gsc1", crs="EPSG:27700", tile_m=hz.TILE_M, cell_m=hz.CELL_M, months=12,
                 quantities=QUANTITIES, scale=SCALE, nodata=NODATA_U8,
                 value="share of the open-sky monthly base = value / scale; multiply by base[quantity][month]",
                 tiles=entries, generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **meta)
    with open(os.path.join(out, "sun-cells.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
    return index


# ---------------------------------------------------------------- the pair
def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.split()[0])
    except Exception:
        return None


def make_guard(t0, seconds, hot=80):
    state = dict(pauses=0, max_temp=None)

    def guard():
        while True:
            t = gpu_temp()
            if t is not None:
                state["max_temp"] = max(state["max_temp"] or 0, t)
            if time.perf_counter() - t0 > seconds:
                raise TimeoutError(f"over the {seconds} s budget")
            if t is None or t < hot:
                return
            state["pauses"] += 1; time.sleep(15)
    return guard, state


def run(hgrid, tmy, offset_h, lat, lon, conv, alt_km, use_gpu=True, witness_cells=256, seed=20260926,
        seconds=300, oe=0, on=0):
    """hgrid (rows, cols, 32) horizon degrees, NaN unknown. Returns (receipt, electron values, bases, nodata)."""
    t0 = time.perf_counter()
    gpu = use_gpu and cp is not None
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if gpu else "cpu (numpy)"
    guard, gstate = make_guard(t0, seconds) if gpu else (None, {})
    rows, cols = hgrid.shape[:2]
    hzc = hgrid.reshape(-1, NAZ)
    S = electron_samples(tmy, offset_h, lat, lon, conv, alt_km)
    P = positron_samples(tmy, offset_h, lat, lon, conv, alt_km)
    flat = np.full((1, NAZ), np.nan)
    e_base = electron_cpu(flat, S)[0]; p_base = positron_direct(flat, P)[0]
    t1 = time.perf_counter()
    e = electron_gpu(hzc, S, guard=guard) if gpu else electron_cpu(hzc, S)
    t2 = time.perf_counter()
    xp = cp if gpu else np
    sky = sky_map(P)
    p = positron(xp, hzc, sky, guard=guard)
    p = cp.asnumpy(p) if gpu else p
    t3 = time.perf_counter()
    # nodata: an unknown horizon in a sector the sun passes through
    sectors = np.unique(np.concatenate([np.floor(S["f"]).astype(int) % NAZ, (np.floor(S["f"]).astype(int) + 1) % NAZ]))
    nodata = np.isnan(hzc[:, sectors]).any(1)
    share = lambda v, b: np.where(b > 0, v / np.where(b > 0, b, 1), 1.0)
    es, ps = share(e, e_base[None]), share(p, p_base[None])
    diff = np.where(nodata[:, None, None], 0.0, ps - es)
    photon = np.abs(diff) > TOL_SHARE
    per_cell = photon.sum((1, 2))
    hot = np.argsort(per_cell)[::-1][:8]
    ok = ~nodata
    # the two suns
    doy, hr, _ = hour_table(tmy, offset_h)
    a1, e1 = noaa(doy, hr, lat, lon); a2, e2 = michalsky(doy.astype(float), hr, lat, lon)
    up = (e1 > 0) | (e2 > 0)
    daz = np.abs((a1 - a2 + 180) % 360 - 180)[up & (e1 > 5)]
    rec = dict(
        device=device, cells=int(len(hzc)), cell_m=hz.CELL_M, months=12, quantities=QUANTITIES,
        nodata_cells=int(nodata.sum()), sun_sectors_used=[int(s) for s in sectors],
        electron=dict(method="hourly sun (NOAA/Spencer), 32-azimuth linear horizon, CUDA kernel", sun_hours=int(len(S["el"])),
                      base_annual=[round(float(v), 3) for v in e_base.sum(0)],
                      mean_share=[round(float(es[ok][..., q].sum() / max(1, ok.sum() * 12)), 5) for q in range(NQ)],
                      seconds=round(t2 - t1, 3)),
        positron=dict(method=f"{SUB} sub-hourly suns (Michalsky 1988), {BIN_DEG} deg sky-map, suffix sums",
                      samples=int(len(P["el"])), bins=len(sky),
                      base_annual=[round(float(v), 3) for v in p_base.sum(0)],
                      mean_share=[round(float(ps[ok][..., q].sum() / max(1, ok.sum() * 12)), 5) for q in range(NQ)],
                      seconds=round(t3 - t2, 3)),
        sun_pair=dict(max_elevation_diff_deg=round(float(np.abs(e1 - e2)[up].max()), 4),
                      max_azimuth_diff_deg_above_5=round(float(daz.max()), 4) if daz.size else None),
        base_relative_diff=[round(float((p_base.sum(0)[q] - e_base.sum(0)[q]) / e_base.sum(0)[q]), 5)
                            if e_base.sum(0)[q] else 0.0 for q in range(NQ)],
        tol_share=TOL_SHARE, pairs=int(photon.size), photons=int(photon.sum()),
        photons_by_quantity=[int(photon[..., q].sum()) for q in range(NQ)],
        photon_cells=int((per_cell > 0).sum()),
        max_abs_share_diff=round(float(np.abs(diff).max()), 4),
        mean_signed_share_diff=[round(float(diff[ok][..., q].mean()), 5) if ok.any() else 0.0 for q in range(NQ)],
        hot_cells=[dict(e=int(oe + (k % cols) * hz.CELL_M), n=int(on + (k // cols) * hz.CELL_M), photons=int(per_cell[k]))
                   for k in hot if per_cell[k]],
        thermal=gstate)
    if witness_cells:
        rng = np.random.default_rng(seed)
        pick = np.unique(np.concatenate([[0, len(hzc) - 1], rng.choice(len(hzc), min(witness_cells, len(hzc)), replace=False)]))
        we = electron_cpu(hzc[pick], S); wp = positron_direct(hzc[pick], P)
        de = np.abs(we - e[pick]) / np.maximum(1.0, np.abs(we)); dp = np.abs(wp - p[pick]) / np.maximum(1.0, np.abs(wp))
        rec["witness"] = dict(cells=int(len(pick)), seed=seed, electron_max_rel_diff=float(de.max()),
                              positron_max_rel_diff=float(dp.max()),
                              photons=int((de > TOL_WIT).sum() + (dp > TOL_WIT).sum()))
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    return rec, e.reshape(rows, cols, 12, NQ), e_base, p_base, nodata.reshape(rows, cols)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True)
    ap.add_argument("--out", help="default SITE/sun/cells")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--witness", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=300)
    a = ap.parse_args(argv)
    clim = json.load(open(os.path.join(a.site, "sun", "sun-climate.json"), encoding="utf-8"))
    blob = open(os.path.join(a.site, "sun", "tmy-hourly.bin"), "rb").read()
    if hashlib.sha256(blob).hexdigest() != clim["hourly"]["sha256"]:
        raise SystemExit("tmy-hourly.bin sha256 does not match sun-climate.json")
    tmy, off = sd.decode_tmy(blob)
    hgrid, hidx = load_horizon(os.path.join(a.site, "horizon"))
    oe, on = hidx["site"]["origin_e"], hidx["site"]["origin_n"]
    alt_km = float(clim["source"]["pvgis_elevation_m"]) / 1000
    rec, e, e_base, p_base, nodata = run(hgrid, tmy, off, clim["latitude"], clim["longitude"],
                                         clim["grid_azimuth_of_true_north_deg"], alt_km, use_gpu=not a.cpu,
                                         witness_cells=a.witness, seconds=a.seconds, oe=oe, on=on)
    out = a.out or os.path.join(a.site, "sun", "cells")
    q = quantise(e, e_base[None, None], nodata)
    meta = dict(site=dict(name=os.path.basename(os.path.normpath(a.site)), origin_e=int(oe), origin_n=int(on)),
                base=dict(units=["h", "kWh/m2", "h", "kWh/m2"],
                          **{QUANTITIES[k]: [round(float(v), 3) for v in e_base[:, k]] for k in range(NQ)}),
                plane="horizontal", terrain="bare-earth DTM horizon (horizon_tiles.py), no vegetation or buildings",
                sun="NOAA general solar position, hourly at HH:00 + PVGIS offset, UTC, reference year 2023",
                source=dict(horizon_index_sha256=hashlib.sha256(open(os.path.join(a.site, "horizon", hz.INDEX), "rb").read()).hexdigest(),
                            sun_climate_sha256=hashlib.sha256(open(os.path.join(a.site, "sun", "sun-climate.json"), "rb").read()).hexdigest()),
                attribution=[clim["attribution"], "© Environment Agency copyright and/or database right 2022. All rights reserved."])
    index = write_tiles(out, q, oe, on, meta)
    rec.update(tiles=len(index["tiles"]), tile_bytes=sum(t["bytes"] for t in index["tiles"]),
               script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               geom_sha256=hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sun_geom.py"), "rb").read()).hexdigest(),
               index_sha256=hashlib.sha256(open(os.path.join(out, "sun-cells.json"), "rb").read()).hexdigest())
    with open(os.path.join(out, "sun_cells_receipt.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("hot_cells", "sun_sectors_used")}, indent=1))
    return 0 if rec.get("witness", {}).get("photons", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
