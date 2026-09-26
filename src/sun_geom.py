"""Sun positions for sun_cells.py: two independent algorithms and the per-hour / sub-hourly sample tables.

noaa       NOAA general solar position (Spencer 1971 Fourier series), one position per TMY hour (electron).
michalsky  Michalsky (1988) Astronomical Almanac algorithm, SUB positions per hour (positron).
clear_dni  Meinel and Meinel (1976) clear-sky DNI with the Laue (1970) altitude term, Kasten and Young (1989) air mass.

Split from sun_cells.py to keep every script under 400 lines; sun_cells_receipt.json carries both hashes.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horizon_tiles as hz  # noqa: E402
import sun_data as sd  # noqa: E402

NAZ = hz.NAZ
SUB = 10                      # positron sub-steps per hour
BIN_DEG = 0.5
NBIN = int(360 / BIN_DEG)
REF_YEAR = 2023
MONTH_START = np.concatenate([[0], np.cumsum(sd.MONTH_DAYS)])     # day-of-year (0-based) of each month start


# ---------------------------------------------------------------- sun position, two algorithms
def noaa(doy, hour_utc, lat, lon):
    """NOAA general solar position (Spencer 1971 Fourier series). doy 1-based, hour decimal UTC.
    Returns (true azimuth clockwise from north, elevation), degrees, geometric."""
    g = 2 * np.pi / 365 * (doy - 1 + (hour_utc - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour_utc * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    el = np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))
    az = np.arctan2(-np.cos(dec) * np.sin(ha), np.sin(dec) * np.cos(la) - np.cos(dec) * np.cos(ha) * np.sin(la))
    return np.degrees(az) % 360, np.degrees(el)


def michalsky(doy, hour_utc, lat, lon, year=REF_YEAR):
    """Michalsky (1988) 'The Astronomical Almanac's algorithm for approximate solar position (1950-2050)',
    Solar Energy 40(3), 227-235. Returns (true azimuth clockwise from north, elevation), degrees, geometric."""
    delta = year - 1949; leap = delta // 4
    jd = 2432916.5 + delta * 365 + leap + doy + hour_utc / 24
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360); gm = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(gm) + np.radians(0.020) * np.sin(2 * gm)
    eps = np.radians(23.439 - 0.0000004 * n)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    gmst = (6.697375 + 0.0657098242 * n + hour_utc) % 24
    lmst = (gmst + lon / 15) % 24
    ha = np.radians(lmst * 15) - ra
    ha = (ha + np.pi) % (2 * np.pi) - np.pi
    la = np.radians(lat)
    el = np.arcsin(np.sin(dec) * np.sin(la) + np.cos(dec) * np.cos(la) * np.cos(ha))
    az = np.arctan2(-np.sin(ha), np.tan(dec) * np.cos(la) - np.sin(la) * np.cos(ha))
    return np.degrees(az) % 360, np.degrees(el)


def clear_dni(el_deg, doy, alt_km):
    """W/m2. Meinel: 1353 * 0.7^(AM^0.678); Laue altitude term a = 0.14 per km; eccentricity 1 + 0.033 cos."""
    z = 90.0 - el_deg
    am = 1.0 / (np.cos(np.radians(z)) + 0.50572 * np.power(np.maximum(96.07995 - z, 1e-6), -1.6364))
    e0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    return np.where(el_deg > 0, 1353.0 * e0 * ((1 - 0.14 * alt_km) * 0.7 ** (am ** 0.678) + 0.14 * alt_km), 0.0)


def hour_table(tmy, offset_h):
    """For each of the 8760 TMY rows: (doy 1-based, decimal UTC hour of the representative instant, month 0..11)."""
    i = np.arange(sd.HOURS)
    doy = i // 24 + 1
    month = np.searchsorted(MONTH_START, doy - 1, side="right") - 1
    return doy, (i % 24) + offset_h, month


def electron_samples(tmy, offset_h, lat, lon, conv, alt_km):
    """One sun position per hour. Returns dict of arrays over sun-up hours (f = fractional azimuth index)."""
    doy, hr, month = hour_table(tmy, offset_h)
    az, el = noaa(doy, hr, lat, lon)
    up = el > 0
    gaz = (az + conv) % 360
    dni = tmy[:, 1]
    s = np.sin(np.radians(el))
    return dict(f=(gaz / (360.0 / NAZ))[up], gaz=gaz[up], el=el[up], month=month[up].astype(np.int32),
                w=np.stack([np.ones(up.sum()), clear_dni(el, doy, alt_km)[up] * s[up] / 1000,
                            (dni[up] >= sd.SUNSHINE_WM2).astype(np.float64), dni[up] * s[up] / 1000], 1))


def positron_samples(tmy, offset_h, lat, lon, conv, alt_km):
    """SUB sun positions per hour over [HH + offset - 0.5, HH + offset + 0.5), weight 1/SUB hour each."""
    doy, hr, month = hour_table(tmy, offset_h)
    k = (np.arange(SUB) + 0.5) / SUB - 0.5
    t = hr[:, None] + k[None, :]
    d = np.repeat(doy[:, None], SUB, 1).astype(np.float64)
    az, el = michalsky(d, t, lat, lon)                 # day rolls over naturally through the hour term
    up = el > 0
    gaz = (az + conv) % 360
    dni = np.repeat(tmy[:, 1][:, None], SUB, 1)
    s = np.sin(np.radians(el))
    m = np.repeat(month[:, None], SUB, 1)
    w = np.stack([np.full(up.sum(), 1.0 / SUB), clear_dni(el, d, alt_km)[up] * s[up] / 1000 / SUB,
                  (dni[up] >= sd.SUNSHINE_WM2) / SUB, dni[up] * s[up] / 1000 / SUB], 1)
    b = np.minimum((gaz[up] / BIN_DEG).astype(np.int64), NBIN - 1)
    return dict(gaz=gaz[up], el=el[up], month=m[up].astype(np.int64), bin=b, w=w)
