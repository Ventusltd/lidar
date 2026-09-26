# SPDX-License-Identifier: Apache-2.0
"""Electron channel of horizon_tiles.py: the ray march, on the GPU (CUDA) and on the CPU (the witness).

Near field (0 to NEAR_M = 32 m): exact on the bilinear surface. The ray is cut where it crosses a grid row
or column line (distances k / |u|), so each piece lies in one cell. Along a piece the bilinear surface is a
quadratic h(d) = A + B d + C d^2, and the tangent s(d) = (h(d) - z0) / d is largest at the piece's end or
where C d^2 = A - z0; both are taken, plus the limit B as d -> 0 on the first piece. A 1 m march missed
knolls within 5 m by up to 3.8 degrees (tester 3, 26 Sept); this has no sampling step there at all.
Far field (33 m on): one bilinear sample every metre, as before.
"""
import math
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the CPU path runs alone
    cp = None

NEAR_M = 32

KERNEL = r"""
extern "C" __global__ void march(const double* z, const int H, const int W, const int* cr, const int* cc,
    const int ncell, const double* ux, const double* uy, const int naz, double* best, int* reach,
    int* at, unsigned long long* samples) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= ncell * naz) return;
  int k = t / naz, a = t % naz;
  double x0 = cc[k], y0 = cr[k], z0 = z[cr[k] * W + cc[k]], m = -1.0e300, vx = ux[a], vy = uy[a];
  double ax = fabs(vx), ay = fabs(vy), d0 = 0.0;
  int last = 0, where = 0, out = 0, kx = 1, ky = 1; unsigned long long n = 0;
  while (d0 < NEAR_M) {
    double nx = ax > 1e-12 ? kx / ax : 1e300, ny = ay > 1e-12 ? ky / ay : 1e300;
    double d1 = fmin(fmin(nx, ny), (double)NEAR_M);
    if (d1 == nx) kx++;
    if (d1 == ny) ky++;
    double dm = 0.5 * (d0 + d1), xm = x0 + dm * vx, ym = y0 + dm * vy;
    if (xm < 0.0 || ym < 0.0 || xm > W - 1 || ym > H - 1) { out = 1; break; }
    int i = (int)floor(xm), j = (int)floor(ym);
    if (i > W - 2) i = W - 2; if (j > H - 2) j = H - 2;
    double z00 = z[j * W + i], z01 = z[j * W + i + 1], z10 = z[(j + 1) * W + i], z11 = z[(j + 1) * W + i + 1];
    n++; last = (int)d1;
    if (!(isnan(z00) || isnan(z01) || isnan(z10) || isnan(z11))) {
      double px = x0 - i, py = y0 - j, b = z01 - z00, c = z10 - z00, e = z00 - z01 - z10 + z11;
      double A = z00 + b * px + c * py + e * px * py, B = b * vx + c * vy + e * (px * vy + py * vx), C = e * vx * vy;
      double s = (A + (B + C * d1) * d1 - z0) / d1;
      if (s > m) { m = s; where = (int)d1; }
      if (d0 == 0.0 && B > m) { m = B; where = 0; }
      if (C != 0.0) {
        double q = (A - z0) / C;
        if (q > 0.0) {
          double ds = sqrt(q);
          if (ds > d0 && ds < d1) {
            s = (A + (B + C * ds) * ds - z0) / ds;
            if (s > m) { m = s; where = (int)ds; }
          }
        }
      }
    }
    d0 = d1;
  }
  for (int d = NEAR_M + 1; !out; d++) {
    double x = x0 + d * vx, y = y0 + d * vy;
    if (x < 0.0 || y < 0.0 || x > W - 1 || y > H - 1) break;
    int i = (int)floor(x), j = (int)floor(y);
    if (i > W - 2) i = W - 2; if (j > H - 2) j = H - 2;
    double fx = x - i, fy = y - j;
    double h = (1 - fx) * (1 - fy) * z[j * W + i] + fx * (1 - fy) * z[j * W + i + 1]
             + (1 - fx) * fy * z[(j + 1) * W + i] + fx * fy * z[(j + 1) * W + i + 1];
    n++; last = d;
    if (isnan(h)) continue;
    double s = (h - z0) / d;
    if (s > m) { m = s; where = d; }
  }
  best[t] = m; reach[t] = last; at[t] = where;
  atomicAdd(samples, n);
}
"""
_kernel = None


def directions(naz):
    a = np.radians(np.arange(naz) * (360.0 / naz))
    return np.sin(a), np.cos(a)                # (east, north) per azimuth


def electron_gpu(z, cr, cc, naz=32):
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(f"#define NEAR_M {NEAR_M}\n" + KERNEL, "march", options=("--fmad=false",))
    H, W = z.shape
    ux, uy = directions(naz)
    zd = cp.asarray(z, cp.float64)
    best = cp.empty(len(cr) * naz, cp.float64); reach = cp.empty(len(cr) * naz, cp.int32)
    samples = cp.zeros(1, cp.uint64); at = cp.empty(len(cr) * naz, cp.int32)
    n = len(cr) * naz; block = 256
    _kernel(((n + block - 1) // block,), (block,), (zd, np.int32(H), np.int32(W), cp.asarray(cr, cp.int32),
            cp.asarray(cc, cp.int32), np.int32(len(cr)), cp.asarray(ux), cp.asarray(uy), np.int32(naz),
            best, reach, at, samples))
    return best.reshape(-1, naz), reach.reshape(-1, naz), at.reshape(-1, naz), int(samples.get()[0])


def _near_cpu(z, r, c, vx, vy):
    """Scalar Python, same pieces as the kernel. Returns (best, at, last, out)."""
    H, W = z.shape
    ax, ay, d0, kx, ky = abs(vx), abs(vy), 0.0, 1, 1
    m, where, last, z0 = -1.0e300, 0, 0, float(z[r, c])
    while d0 < NEAR_M:
        nx = kx / ax if ax > 1e-12 else 1e300
        ny = ky / ay if ay > 1e-12 else 1e300
        d1 = min(nx, ny, float(NEAR_M))
        kx += d1 == nx; ky += d1 == ny
        dm = 0.5 * (d0 + d1); xm, ym = c + dm * vx, r + dm * vy
        if xm < 0 or ym < 0 or xm > W - 1 or ym > H - 1:
            return m, where, last, True
        i, j = min(math.floor(xm), W - 2), min(math.floor(ym), H - 2)
        q00, q01, q10, q11 = (float(v) for v in (z[j, i], z[j, i + 1], z[j + 1, i], z[j + 1, i + 1]))
        last = int(d1)
        if not any(math.isnan(v) for v in (q00, q01, q10, q11)):
            px, py = c - i, r - j
            b, cy, e = q01 - q00, q10 - q00, q00 - q01 - q10 + q11
            A = q00 + b * px + cy * py + e * px * py
            B = b * vx + cy * vy + e * (px * vy + py * vx); C = e * vx * vy
            s = (A + (B + C * d1) * d1 - z0) / d1
            if s > m:
                m, where = s, int(d1)
            if d0 == 0.0 and B > m:
                m, where = B, 0
            if C != 0.0 and (A - z0) / C > 0.0:
                ds = math.sqrt((A - z0) / C)
                if d0 < ds < d1:
                    s = (A + (B + C * ds) * ds - z0) / ds
                    if s > m:
                        m, where = s, int(ds)
        d0 = d1
    return m, where, last, False


def electron_cpu(z, r, c, naz=32):
    """One cell, all azimuths, on the CPU: exact near field, then bilinear as two lerps (x, then y)."""
    H, W = z.shape
    ux, uy = directions(naz)
    best = np.full(naz, -1.0e300); reach = np.zeros(naz, np.int64); at = np.zeros(naz, np.int64)
    dmax = int(np.ceil(np.hypot(H, W))) + 2
    d = np.arange(NEAR_M + 1, max(dmax, NEAR_M + 1), dtype=np.float64)
    for a in range(naz):
        best[a], at[a], reach[a], out = _near_cpu(z, r, c, float(ux[a]), float(uy[a]))
        if out:
            continue
        x = c + d * ux[a]; y = r + d * uy[a]
        inside = (x >= 0) & (y >= 0) & (x <= W - 1) & (y <= H - 1)
        stop = np.argmin(inside) if not inside.all() else len(d)
        x, y, dd = x[:stop], y[:stop], d[:stop]
        if not len(dd):
            continue
        i = np.minimum(np.floor(x).astype(np.int64), W - 2); j = np.minimum(np.floor(y).astype(np.int64), H - 2)
        fx, fy = x - i, y - j
        lo = z[j, i] + (z[j, i + 1] - z[j, i]) * fx
        hi = z[j + 1, i] + (z[j + 1, i + 1] - z[j + 1, i]) * fx
        s = (lo + (hi - lo) * fy - z[r, c]) / dd
        if np.isfinite(s).any() and np.nanmax(s) > best[a]:
            k = int(np.nanargmax(s)); best[a] = s[k]; at[a] = int(dd[k])
        reach[a] = int(dd[-1])
    return best, reach, at
