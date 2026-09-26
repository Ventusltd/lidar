# SPDX-License-Identifier: Apache-2.0
"""Ponding hollows for flow_tiles.py: the real depressions of the fill, written so a layer can show them.

A hollow is an 8-connected group of cells where the flat fill stands above the ground (depth = W - Z > 0).
Tiny ones are everywhere on a 1 m DTM (24,894 on open-land-01, most a few litres), so only hollows of at
least HOLLOW_MIN_M3 or HOLLOW_MIN_DEPTH_M are written; the receipt still counts all of them.

flow-hollows.json (LF, UTF-8): every kept hollow with its id, water level, volume, area, deepest point,
centroid, bounding box and tiles, plus the attribution and licence. Per tile with a kept hollow,
hollows/<key>.gph:
    .gph v1 ("GGP1"), little-endian, 32-byte header then one u8 body:
    0 char[4] "GGP1" | 4 u16 version=1 | 6 u16 samples=257 | 8 u16 spacing_mm | 10 u16 unit_mm=10
   12 i32 origin_e_m | 16 i32 origin_n_m (SW corner) | 20 u32 wet_count | 24 u32 hollows | 28 u32 reserved=0
   32 u8[257*257] depth in whole centimetres (1..254, 254 = 2.54 m or more) inside a kept hollow;
      0 dry or in a hollow too small to keep; 255 no data. Rows SOUTH to NORTH, each WEST to EAST.
"""
import hashlib, json, os, struct
import numpy as np

HOLLOW_MIN_M3 = 1.0
HOLLOW_MIN_DEPTH_M = 0.1
MAGIC = b"GGP1"
HEADER = struct.Struct("<4sHHHHiiIII")
assert HEADER.size == 32
INDEX = "flow-hollows.json"


def hollows(grid, Wf, nod, sp, label8):
    """(labels of kept hollows (0 = none), list of hollow dicts in site-local cells)."""
    depth = np.where(nod, 0.0, Wf - grid)
    lab, n = label8(depth > 0)
    if n == 0:
        return np.zeros(grid.shape, np.int32), [], depth
    flat = lab.ravel(); d = depth.ravel()
    vol = np.bincount(flat, weights=d, minlength=n + 1) * sp * sp
    mx = np.zeros(n + 1); np.maximum.at(mx, flat, d)
    keep = (vol >= HOLLOW_MIN_M3) | (mx >= HOLLOW_MIN_DEPTH_M); keep[0] = False
    new = np.zeros(n + 1, np.int32); new[keep] = np.arange(1, int(keep.sum()) + 1)
    kl = new[lab]
    out = []
    order = np.argsort(flat, kind="stable")
    start = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=n + 1))])
    for k in np.nonzero(keep)[0]:
        cells = order[start[k]:start[k + 1]]
        rr, cc = np.divmod(cells, grid.shape[1])
        i = int(np.argmax(d[cells]))
        out.append(dict(id=int(new[k]), level_m=round(float(Wf[rr[0], cc[0]]), 3), volume_m3=round(float(vol[k]), 3),
                        area_m2=round(len(rr) * sp * sp, 1), max_depth_m=round(float(mx[k]), 3),
                        deepest=(int(rr[i]), int(cc[i])), centroid=(float(rr.mean()), float(cc.mean())),
                        rows=(int(rr.min()), int(rr.max())), cols=(int(cc.min()), int(cc.max()))))
    return kl, out, depth


def encode(depth_cm, e0, n0, spacing_mm, wet, count):
    head = HEADER.pack(MAGIC, 1, depth_cm.shape[0], spacing_mm, 10, int(e0), int(n0), int(wet), int(count), 0)
    return head + np.ascontiguousarray(depth_cm, np.uint8).tobytes()


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError("short .gph blob")
    magic, ver, n, sp, unit, e0, n0, wet, count, _ = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != 1 or len(blob) != HEADER.size + n * n:
        raise ValueError("bad .gph blob")
    head = dict(samples=n, spacing_mm=sp, unit_mm=unit, origin_e_m=e0, origin_n_m=n0, wet_count=wet, hollows=count)
    return head, np.frombuffer(blob, np.uint8, offset=HEADER.size).reshape(n, n)


def write(out_dir, o, oe, on, spacing_mm, tile_m, notice, label8):
    """Write flow-hollows.json and hollows/<key>.gph; returns the index dict."""
    sp = spacing_mm / 1000; N = tile_m + 1
    kl, hs, depth = hollows(o["grid"], o["Wf"], o["nod"], sp, label8)
    cm = np.clip(np.rint(depth * 100), 1, 254).astype(np.uint8)
    body = np.where(kl > 0, cm, 0).astype(np.uint8); body[o["nod"]] = 255
    rows, cols = body.shape
    tiles = []
    os.makedirs(os.path.join(out_dir, "hollows"), exist_ok=True)
    for iy in range((rows - 1) // tile_m):
        for ix in range((cols - 1) // tile_m):
            r0, c0 = iy * tile_m, ix * tile_m
            t = body[r0:r0 + N, c0:c0 + N]; ids = np.unique(kl[r0:r0 + N, c0:c0 + N]); ids = ids[ids > 0]
            if not len(ids):
                continue
            key = f"{ix}_{iy}"; rel = f"hollows/{key}.gph"
            blob = encode(t, oe + c0 * sp, on + r0 * sp, spacing_mm, int(((t > 0) & (t < 255)).sum()), len(ids))
            with open(os.path.join(out_dir, rel), "wb") as f:
                f.write(blob)
            tiles.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(oe + c0 * sp),
                              n0=int(on + r0 * sp), bytes=len(blob), hollows=ids.tolist()))
            for h in hs:
                if h["id"] in ids:
                    h.setdefault("tiles", []).append(key)
    for h in hs:
        (r, c), (cr, ccn) = h.pop("deepest"), h.pop("centroid")
        (r0, r1), (c0, c1) = h.pop("rows"), h.pop("cols")
        h.update(deepest_e=oe + c * sp, deepest_n=on + r * sp, centroid_e=round(oe + ccn * sp, 1),
                 centroid_n=round(on + cr * sp, 1), bbox=[oe + c0 * sp, on + r0 * sp, oe + c1 * sp, on + r1 * sp])
    index = dict(format="gph1", crs="EPSG:27700", tile_m=tile_m, spacing_m=sp, keep_min_m3=HOLLOW_MIN_M3,
                 keep_min_depth_m=HOLLOW_MIN_DEPTH_M,
                 values="u8 depth in cm (1..254) inside a kept hollow; 0 dry or too small to keep; 255 no data",
                 meaning="water that would stand in a closed hollow of the bare-earth DTM (flat fill minus ground); "
                         "computed from LiDAR, not an official flood map",
                 **notice, hollows=hs, tiles=tiles)
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return index
