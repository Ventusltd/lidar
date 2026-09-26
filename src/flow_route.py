"""Flow routing core for flow_tiles.py: constants, CUDA kernels, the GPU path and the CPU witness path.

Split out of flow_tiles.py to keep each script under 400 lines; the method, the pair and the tile format
are described there. fill: raster-sweep relaxation (Planchon and Darboux 2002) on the GPU against
priority-flood (Barnes et al. 2014) on the CPU; routing D8 against D-infinity (Tarboton 1997).
"""
import heapq, struct, time
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the CPU witness path still runs (small grids only)
    cp = None

TILE_M = 256
N = TILE_M + 1
EPS = 1e-5                     # metres per step across a filled flat (routing surface only)
CHANNEL_M2 = 2000.0
LOG_SCALE = 10
NODATA = 255
OUTLET = 8
SQRT2 = 1.4142135623730951
PI4 = 0.7853981633974483
DR = (1, 1, 0, -1, -1, -1, 0, 1)       # N NE E SE S SW W NW, rows run south to north
DC = (0, 1, 1, 1, 0, -1, -1, -1)
FACETS = ((2, 1), (0, 1), (0, 7), (6, 7), (6, 5), (4, 5), (4, 3), (2, 3))   # (cardinal, diagonal)
MAX_ROUNDS = 4000              # fill: rounds of four sweeps
MAX_LEVELS = 400000            # accumulation: peel levels
MAX_SECONDS = 120.0
BATCH = 64
TOL_WIT = 1e-9
MAGIC = b"GGF1"
HEADER = struct.Struct("<4sHHHHiiBBHII")
assert HEADER.size == 32
INDEX = "flow-tiles.json"
RECEIPT = "flow_receipt.json"

KSRC = r"""
#define SQRT2 1.4142135623730951
#define PI4 0.7853981633974483
__constant__ int DR[8] = {1, 1, 0, -1, -1, -1, 0, 1};
__constant__ int DC[8] = {0, 1, 1, 1, 0, -1, -1, -1};
__constant__ int F1[8] = {2, 0, 0, 6, 6, 4, 4, 2};
__constant__ int F2[8] = {1, 1, 7, 7, 5, 5, 3, 3};

extern "C" __global__ void sweep(double* W, const double* Z, const unsigned char* fixed,
                                 int R, int C, int dir, double eps, int* changed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int L = dir < 2 ? R : C, T = dir < 2 ? C : R;
    if (t >= T) return;
    int any = 0;
    for (int k = 0; k < L; k++) {
        int kk = (dir == 0 || dir == 2) ? k : L - 1 - k;
        int r = dir < 2 ? kk : t, c = dir < 2 ? t : kk;
        int i = r * C + c;
        if (fixed[i]) continue;
        double w = W[i], z = Z[i];
        if (w == z) continue;
        double m = W[i + C];
        for (int q = 1; q < 8; q++) m = fmin(m, W[(r + DR[q]) * C + c + DC[q]]);
        double nw = (z > m) ? z : m + eps;
        if (nw < w) { W[i] = nw; any = 1; }
    }
    if (any) atomicAdd(changed, 1);
}

extern "C" __global__ void d8(const double* W, const unsigned char* fixed, const unsigned char* nod,
                              int R, int C, double sp, unsigned char* dir, int* tgt, double* frac,
                              double* step, int* sinks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * C) return;
    tgt[2 * i] = -1; tgt[2 * i + 1] = -1; frac[2 * i] = 0; frac[2 * i + 1] = 0; step[2 * i] = 0; step[2 * i + 1] = 0;
    if (nod[i]) { dir[i] = 255; return; }
    if (fixed[i]) { dir[i] = 8; return; }
    int r = i / C, c = i % C;
    double w = W[i], best = 0.0; int bk = -1, bj = -1;
    for (int k = 0; k < 8; k++) {
        int j = (r + DR[k]) * C + c + DC[k];
        if (nod[j]) continue;
        double d = (k & 1) ? sp * SQRT2 : sp;
        double drop = (w - W[j]) / d;
        if (drop > best) { best = drop; bk = k; bj = j; }
    }
    if (bk < 0) { dir[i] = 8; atomicAdd(sinks, 1); return; }
    dir[i] = (unsigned char)bk; tgt[2 * i] = bj; frac[2 * i] = 1.0;
    step[2 * i] = (bk & 1) ? sp * SQRT2 : sp;
}

extern "C" __global__ void dinf(const double* W, const unsigned char* fixed, const unsigned char* nod,
                                int R, int C, double sp, int* tgt, double* frac, int* sinks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * C) return;
    tgt[2 * i] = -1; tgt[2 * i + 1] = -1; frac[2 * i] = 0; frac[2 * i + 1] = 0;
    if (nod[i] || fixed[i]) return;
    int r = i / C, c = i % C;
    double w = W[i], best = 0.0, br = 0.0; int b1 = -1, b2 = -1;
    for (int f = 0; f < 8; f++) {
        int j1 = (r + DR[F1[f]]) * C + c + DC[F1[f]], j2 = (r + DR[F2[f]]) * C + c + DC[F2[f]];
        if (nod[j1] || nod[j2]) continue;
        double s1 = (w - W[j1]) / sp, s2 = (W[j1] - W[j2]) / sp;
        double rr = atan2(s2, s1), s;
        if (rr < 0.0) { rr = 0.0; s = s1; }
        else if (rr > PI4) { rr = PI4; s = (w - W[j2]) / (sp * SQRT2); }
        else s = sqrt(s1 * s1 + s2 * s2);
        if (s > best) { best = s; br = rr; b1 = j1; b2 = j2; }
    }
    if (b1 < 0) { atomicAdd(sinks, 1); return; }
    double a2 = br / PI4, a1 = 1.0 - a2;
    if (a1 > 0.0) { tgt[2 * i] = b1; frac[2 * i] = a1; }
    if (a2 > 0.0) { tgt[2 * i + 1] = b2; frac[2 * i + 1] = a2; }
}

extern "C" __global__ void indegree(const int* tgt, int n, int* indeg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n) return;
    if (tgt[i] >= 0) atomicAdd(&indeg[tgt[i]], 1);
}

extern "C" __global__ void peel(const int* tgt, const double* frac, const double* step, int n, int withlen,
                                const int* indeg, int* sub, unsigned char* done, double* acc, double* len,
                                unsigned long long* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || done[i] || indeg[i] != 0) return;
    done[i] = 1;
    atomicAdd(count, 1ULL);
    for (int k = 0; k < 2; k++) {
        int j = tgt[2 * i + k];
        if (j < 0) continue;
        atomicAdd(&acc[j], acc[i] * frac[2 * i + k]);
        atomicAdd(&sub[j], 1);
        if (withlen) {
            double v = len[i] + step[2 * i + k];
            atomicMax((unsigned long long*)&len[j], (unsigned long long)__double_as_longlong(v));
        }
    }
}

extern "C" __global__ void settle(int n, int* indeg, int* sub) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (sub[i]) { indeg[i] -= sub[i]; sub[i] = 0; }
}
"""
_MOD = None


def _k(name):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=KSRC, options=("--fmad=false",))
    return _MOD.get_function(name)


def _launch(name, n, args):
    _k(name)(((n + 255) // 256,), (256,), args)


# ---------------------------------------------------------------- masks shared by both paths
def masks(grid):
    """nod: no data. fixed: outlets whose level is the ground (site edge, cells touching no data)."""
    nod = ~np.isfinite(grid)
    fixed = np.zeros(grid.shape, bool)
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    if nod.any():
        near = nod.copy()
        near[1:, :] |= nod[:-1, :]; near[:-1, :] |= nod[1:, :]
        near2 = near.copy(); near2[:, 1:] |= near[:, :-1]; near2[:, :-1] |= near[:, 1:]
        fixed |= near2
    fixed &= ~nod
    return nod, fixed


# ---------------------------------------------------------------- GPU path
def gpu_fill(z, nod, fixed, eps, deadline):
    R, C = z.shape
    W = cp.where(cp.asarray(fixed), z, cp.inf)
    W[cp.asarray(nod)] = cp.inf
    fx = cp.asarray((fixed | nod).astype(np.uint8))
    changed = cp.zeros(1, cp.int32)
    for rnd in range(1, MAX_ROUNDS + 1):
        changed[0] = 0
        for d in range(4):
            T = C if d < 2 else R
            _launch("sweep", T, (W, z, fx, np.int32(R), np.int32(C), np.int32(d), np.float64(eps), changed))
        if int(changed[0]) == 0:
            return W, rnd
        if time.perf_counter() > deadline:
            break
    raise RuntimeError(f"fill did not converge in {rnd} rounds")


def gpu_route(W, nod, fixed, sp, method):
    R, C = W.shape; n = R * C
    fx = cp.asarray(fixed.astype(np.uint8)); nd = cp.asarray(nod.astype(np.uint8))
    tgt = cp.empty(2 * n, cp.int32); frac = cp.empty(2 * n, cp.float64); sinks = cp.zeros(1, cp.int32)
    if method == "d8":
        dr = cp.empty(n, cp.uint8); step = cp.empty(2 * n, cp.float64)
        _launch("d8", n, (W, fx, nd, np.int32(R), np.int32(C), np.float64(sp), dr, tgt, frac, step, sinks))
        return dict(dir=dr.reshape(R, C), tgt=tgt, frac=frac, step=step, sinks=int(sinks[0]))
    _launch("dinf", n, (W, fx, nd, np.int32(R), np.int32(C), np.float64(sp), tgt, frac, sinks))
    return dict(tgt=tgt, frac=frac, step=None, sinks=int(sinks[0]))


def gpu_accumulate(route, nod, sp, deadline):
    n = nod.size
    tgt, frac, step = route["tgt"], route["frac"], route["step"]
    withlen = step is not None
    indeg = cp.zeros(n, cp.int32); sub = cp.zeros(n, cp.int32)
    _launch("indegree", 2 * n, (tgt, np.int32(n), indeg))
    ndf = cp.asarray(nod.ravel())
    done = ndf.astype(cp.uint8)
    acc = cp.where(ndf, 0.0, sp * sp)
    ln = cp.zeros(n, cp.float64)
    count = cp.zeros(1, cp.uint64)
    want = int(n - nod.sum()); last = -1; levels = 0
    stp = step if withlen else frac
    while True:
        for _ in range(BATCH):
            _launch("peel", n, (tgt, frac, stp, np.int32(n), np.int32(withlen), indeg, sub, done, acc, ln, count))
            _launch("settle", n, (np.int32(n), indeg, sub))
        levels += BATCH
        got = int(count[0])
        if got == want:
            break
        if got == last or levels > MAX_LEVELS or time.perf_counter() > deadline:
            raise RuntimeError(f"accumulation stuck: {got}/{want} cells after {levels} levels")
        last = got
    return acc, (ln if withlen else None), levels


# ---------------------------------------------------------------- CPU path (the witness): other code
def cpu_fill(z, nod, fixed, eps):
    """Priority-flood (Barnes et al. 2014) with a heap; eps = 0 gives the flat fill."""
    R, C = z.shape
    zf = z.ravel(); W = np.full(R * C, np.inf)
    seen = (nod | fixed).ravel().copy()
    heap = []
    for i in np.flatnonzero(fixed.ravel()):
        W[i] = zf[i]; heap.append((zf[i], int(i)))
    heapq.heapify(heap)
    offs = [(dr, dc) for dr, dc in zip(DR, DC)]
    while heap:
        w, i = heapq.heappop(heap)
        r, c = divmod(i, C)
        for dr, dc in offs:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= R or cc >= C:
                continue
            j = rr * C + cc
            if seen[j]:
                continue
            seen[j] = True
            W[j] = zf[j] if zf[j] > w else w + eps
            heapq.heappush(heap, (W[j], j))
    return W.reshape(R, C)


def _nb(a, k):
    """Neighbour k of every interior cell, as a view (R-2, C-2)."""
    R, C = a.shape
    return a[1 + DR[k]:R - 1 + DR[k], 1 + DC[k]:C - 1 + DC[k]]


def cpu_route(W, nod, fixed, sp, method):
    R, C = W.shape; n = R * C
    idx = np.arange(n).reshape(R, C)
    inner = ~(fixed | nod)[1:-1, 1:-1]
    w0 = W[1:-1, 1:-1]
    tgt = np.full((R, C, 2), -1, np.int64); frac = np.zeros((R, C, 2)); sinks = 0
    if method == "d8":
        drops = np.stack([np.where(_nb(nod, k), -np.inf, (w0 - _nb(W, k)) / (sp * SQRT2 if k & 1 else sp))
                          for k in range(8)])
        k = drops.argmax(0); best = drops.max(0)
        ok = inner & (best > 0)
        sinks = int((inner & ~(best > 0)).sum())
        dr = np.full((R, C), OUTLET, np.uint8); dr[nod] = NODATA
        dr[1:-1, 1:-1][ok] = k[ok]
        j = np.stack([_nb(idx, q) for q in range(8)])
        jt = np.take_along_axis(j, k[None], 0)[0]
        tgt[1:-1, 1:-1, 0][ok] = jt[ok]; frac[1:-1, 1:-1, 0][ok] = 1.0
        step = np.zeros((R, C, 2)); step[1:-1, 1:-1, 0][ok] = np.where(k[ok] & 1, sp * SQRT2, sp)
        return dict(dir=dr, tgt=tgt.reshape(-1), frac=frac.reshape(-1), step=step.reshape(-1), sinks=sinks)
    best = np.zeros(w0.shape); br = np.zeros(w0.shape); b1 = np.full(w0.shape, -1); b2 = np.full(w0.shape, -1)
    for f1, f2 in FACETS:
        e1, e2 = _nb(W, f1), _nb(W, f2)
        s1 = (w0 - e1) / sp; s2 = (e1 - e2) / sp
        r = np.arctan2(s2, s1)
        s = np.where(r < 0, s1, np.where(r > PI4, (w0 - e2) / (sp * SQRT2), np.sqrt(s1 * s1 + s2 * s2)))
        r = np.clip(r, 0.0, PI4)
        take = (s > best) & ~_nb(nod, f1) & ~_nb(nod, f2)
        best = np.where(take, s, best); br = np.where(take, r, br)
        b1 = np.where(take, _nb(idx, f1), b1); b2 = np.where(take, _nb(idx, f2), b2)
    ok = inner & (b1 >= 0)
    sinks = int((inner & (b1 < 0)).sum())
    a2 = br / PI4; a1 = 1.0 - a2
    t0, t1, f0, f1_ = tgt[1:-1, 1:-1, 0], tgt[1:-1, 1:-1, 1], frac[1:-1, 1:-1, 0], frac[1:-1, 1:-1, 1]
    m1, m2 = ok & (a1 > 0), ok & (a2 > 0)
    t0[m1] = b1[m1]; f0[m1] = a1[m1]; t1[m2] = b2[m2]; f1_[m2] = a2[m2]
    return dict(tgt=tgt.reshape(-1), frac=frac.reshape(-1), step=None, sinks=sinks)


def cpu_accumulate(route, W, nod, sp):
    """Highest level first: every flow edge runs to a strictly lower level, so this is topological."""
    n = nod.size
    tgt, frac, step = route["tgt"], route["frac"], route["step"]
    acc = np.where(nod.ravel(), 0.0, sp * sp); ln = np.zeros(n)
    order = np.argsort(-np.where(nod, -np.inf, W).ravel(), kind="stable")
    tl, fl = tgt.tolist(), frac.tolist(); sl = step.tolist() if step is not None else None
    a = acc.tolist(); L = ln.tolist()
    for i in order.tolist():
        for k in (2 * i, 2 * i + 1):
            j = tl[k]
            if j >= 0:
                a[j] += a[i] * fl[k]
                if sl is not None and L[i] + sl[k] > L[j]:
                    L[j] = L[i] + sl[k]
    return np.array(a), (np.array(L) if step is not None else None), None


# ---------------------------------------------------------------- one pipeline, either path
def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def pipeline(grid, sp, use_gpu, deadline):
    with np.errstate(invalid="ignore", divide="ignore"):
        return _pipeline(grid, sp, use_gpu, deadline)


def _pipeline(grid, sp, use_gpu, deadline):
    grid = np.asarray(grid, np.float64)
    nod, fixed = masks(grid)
    zg = np.where(nod, np.inf, grid)
    t = {}
    if use_gpu:
        z = cp.asarray(zg)
        t0 = time.perf_counter()
        Wf, rf = gpu_fill(z, nod, fixed, 0.0, deadline); We, re = gpu_fill(z, nod, fixed, EPS, deadline)
        cp.cuda.Device().synchronize(); t["fill_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        e = gpu_route(We, nod, fixed, sp, "d8"); p = gpu_route(We, nod, fixed, sp, "dinf")
        accE, lenE, levE = gpu_accumulate(e, nod, sp, deadline)
        accP, _, levP = gpu_accumulate(p, nod, sp, deadline)
        cp.cuda.Device().synchronize(); t["route_s"] = time.perf_counter() - t0
        out = dict(Wf=to_host(Wf), We=to_host(We), dir=to_host(e["dir"]), tgtE=to_host(e["tgt"]),
                   stepE=to_host(e["step"]), accE=to_host(accE).reshape(grid.shape),
                   lenE=to_host(lenE).reshape(grid.shape), accP=to_host(accP).reshape(grid.shape),
                   sinksE=e["sinks"], sinksP=p["sinks"], fill_rounds=[rf, re], levels=[levE, levP])
    else:
        t0 = time.perf_counter()
        Wf = cpu_fill(zg, nod, fixed, 0.0); We = cpu_fill(zg, nod, fixed, EPS)
        t["fill_s"] = time.perf_counter() - t0; t0 = time.perf_counter()
        e = cpu_route(We, nod, fixed, sp, "d8"); p = cpu_route(We, nod, fixed, sp, "dinf")
        accE, lenE, _ = cpu_accumulate(e, We, nod, sp); accP, _, _ = cpu_accumulate(p, We, nod, sp)
        t["route_s"] = time.perf_counter() - t0
        out = dict(Wf=Wf, We=We, dir=e["dir"], tgtE=e["tgt"], stepE=e["step"], accE=accE.reshape(grid.shape),
                   lenE=lenE.reshape(grid.shape), accP=accP.reshape(grid.shape), sinksE=e["sinks"],
                   sinksP=p["sinks"], fill_rounds=None, levels=None)
    out.update(nod=nod, fixed=fixed, grid=grid, times={k: round(v, 3) for k, v in t.items()})
    return out