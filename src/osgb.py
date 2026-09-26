"""WGS84 latitude/longitude <-> British National Grid (EPSG:27700), NumPy, vectorised.

Method (stated because the answer depends on it):
  BNG -> WGS84: Transverse Mercator inverse on the Airy 1830 ellipsoid with the National Grid
  constants (F0 0.9996012717, true origin 49N 2W, false origin E 400000 N -100000), then a
  7-parameter Helmert transformation OSGB36 -> WGS84 with the OSGB36 ellipsoidal height taken as
  zero. WGS84 -> BNG is the same chain run backwards (inverse Helmert matrix, TM forward series).
  This mirrors web/world/bng.mjs in the viewer, so both ends place a tile in the same spot.

Accuracy: the Helmert route agrees with the definitive OSTN15 transformation to about 3.5 m
(Ordnance Survey, "A guide to coordinate systems in Great Britain", v3.6, section 6.6). That is
roughly a tenth of a Copernicus GLO-30 cell, so it is adequate for 30 m data and is NOT adequate
for 1 m data. OSTN15 is not held here.

Sources: Ordnance Survey, "A guide to coordinate systems in Great Britain" v3.6 (2020): Annex B
(ellipsoid <-> Cartesian), section 6.6 (Helmert parameters), Annex C (Transverse Mercator).
"""
import numpy as np

AIRY_A, AIRY_B = 6377563.396, 6356256.909
WGS_A, WGS_B = 6378137.0, 6356752.3141
F0 = 0.9996012717
LAT0, LON0 = np.radians(49.0), np.radians(-2.0)
E0, N0 = 400000.0, -100000.0
SEC = np.radians(1.0 / 3600.0)
# OSGB36 -> WGS84 (the OS guide gives WGS84 -> OSGB36; these are its negatives).
T = np.array([446.448, -125.157, 542.060])
RX, RY, RZ = 0.1502 * SEC, 0.2470 * SEC, 0.8421 * SEC
S = 1.0 - 20.4894e-6
M = np.array([[S, -RZ, RY], [RZ, S, -RX], [-RY, RX, S]])
MI = np.linalg.inv(M)
HELMERT_ACCURACY_M = 3.5


def _e2(a, b):
    return 1.0 - (b * b) / (a * a)


def _to_xyz(lat, lon, h, a, b):
    e2 = _e2(a, b)
    nu = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.stack([(nu + h) * np.cos(lat) * np.cos(lon),
                     (nu + h) * np.cos(lat) * np.sin(lon),
                     ((1.0 - e2) * nu + h) * np.sin(lat)])


def _from_xyz(xyz, a, b):
    x, y, z = xyz
    e2 = _e2(a, b)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(10):  # converges to well under 1e-12 rad in 3 or 4 steps
        nu = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * nu * np.sin(lat), p)
    nu = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - nu
    return lat, np.arctan2(y, x), h


def _meridian_arc(phi):
    n = (AIRY_A - AIRY_B) / (AIRY_A + AIRY_B)
    n2, n3 = n * n, n * n * n
    d, s = phi - LAT0, phi + LAT0
    return AIRY_B * F0 * ((1 + n + 1.25 * n2 + 1.25 * n3) * d
                          - (3 * n + 3 * n2 + 2.625 * n3) * np.sin(d) * np.cos(s)
                          + (1.875 * n2 + 1.875 * n3) * np.sin(2 * d) * np.cos(2 * s)
                          - (35.0 / 24.0) * n3 * np.sin(3 * d) * np.cos(3 * s))


def tm_forward(lat, lon):
    """OSGB36 lat/lon (radians) -> (E, N) metres. OS guide Annex C, equations C1-C3."""
    e2 = _e2(AIRY_A, AIRY_B)
    sl, cl, tl = np.sin(lat), np.cos(lat), np.tan(lat)
    nu = AIRY_A * F0 / np.sqrt(1 - e2 * sl ** 2)
    rho = AIRY_A * F0 * (1 - e2) * (1 - e2 * sl ** 2) ** -1.5
    eta2 = nu / rho - 1
    m = _meridian_arc(lat)
    i = m + N0
    ii = nu / 2 * sl * cl
    iii = nu / 24 * sl * cl ** 3 * (5 - tl ** 2 + 9 * eta2)
    iiia = nu / 720 * sl * cl ** 5 * (61 - 58 * tl ** 2 + tl ** 4)
    iv = nu * cl
    v = nu / 6 * cl ** 3 * (nu / rho - tl ** 2)
    vi = nu / 120 * cl ** 5 * (5 - 18 * tl ** 2 + tl ** 4 + 14 * eta2 - 58 * tl ** 2 * eta2)
    dl = lon - LON0
    n = i + ii * dl ** 2 + iii * dl ** 4 + iiia * dl ** 6
    e = E0 + iv * dl + v * dl ** 3 + vi * dl ** 5
    return e, n


def tm_inverse(e, n):
    """(E, N) metres -> OSGB36 lat/lon (radians). OS guide Annex C, equations C6-C8."""
    e = np.asarray(e, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    e2 = _e2(AIRY_A, AIRY_B)
    lat = (n - N0) / (AIRY_A * F0) + LAT0
    for _ in range(20):  # until N - N0 - M < 0.01 mm everywhere
        resid = n - N0 - _meridian_arc(lat)
        if np.all(np.abs(resid) < 1e-5):
            break
        lat = lat + resid / (AIRY_A * F0)
    sl, cl, tl = np.sin(lat), np.cos(lat), np.tan(lat)
    nu = AIRY_A * F0 / np.sqrt(1 - e2 * sl ** 2)
    rho = AIRY_A * F0 * (1 - e2) * (1 - e2 * sl ** 2) ** -1.5
    eta2 = nu / rho - 1
    vii = tl / (2 * rho * nu)
    viii = tl / (24 * rho * nu ** 3) * (5 + 3 * tl ** 2 + eta2 - 9 * tl ** 2 * eta2)
    ix = tl / (720 * rho * nu ** 5) * (61 + 90 * tl ** 2 + 45 * tl ** 4)
    x = 1 / (cl * nu)
    xi = 1 / (cl * 6 * nu ** 3) * (nu / rho + 2 * tl ** 2)
    xii = 1 / (cl * 120 * nu ** 5) * (5 + 28 * tl ** 2 + 24 * tl ** 4)
    xiia = 1 / (cl * 5040 * nu ** 7) * (61 + 662 * tl ** 2 + 1320 * tl ** 4 + 720 * tl ** 6)
    de = e - E0
    lat_out = lat - vii * de ** 2 + viii * de ** 4 - ix * de ** 6
    lon_out = LON0 + x * de - xi * de ** 3 + xii * de ** 5 - xiia * de ** 7
    return lat_out, lon_out


def bng_to_wgs84(e, n):
    """BNG metres -> WGS84 (lat, lon) in degrees. Arrays of any matching shape."""
    lat, lon = tm_inverse(e, n)
    xyz = _to_xyz(lat, lon, 0.0, AIRY_A, AIRY_B)
    shp = xyz.shape
    w = (M @ xyz.reshape(3, -1)).reshape(shp) + T.reshape((3,) + (1,) * (len(shp) - 1))
    la, lo, _ = _from_xyz(w, WGS_A, WGS_B)
    return np.degrees(la), np.degrees(lo)


def wgs84_to_bng(lat_deg, lon_deg):
    """WGS84 degrees -> BNG metres, by inverting the chain above (Airy height zero, one fixed-point
    pass on the ellipsoidal height), so that it closes against bng_to_wgs84 to well under 1 mm."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    h = np.zeros_like(lat)
    for _ in range(3):  # pick the WGS84 height that lands on zero Airy height
        xyz = _to_xyz(lat, lon, h, WGS_A, WGS_B)
        shp = xyz.shape
        o = (MI @ (xyz.reshape(3, -1) - T.reshape(3, 1))).reshape(shp)
        la, lo, ho = _from_xyz(o, AIRY_A, AIRY_B)
        h = h - ho
    e, n = tm_forward(la, lo)
    # the OS forward and inverse series are truncations that part by ~1 mm far from 2W; one
    # fixed-point step against tm_inverse makes the pair close (as the viewer's bng.mjs does)
    fe, fn = tm_forward(*tm_inverse(e, n))
    return e + (e - fe), n + (n - fn)
