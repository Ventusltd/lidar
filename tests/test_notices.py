"""Tests for the attribution, licence and caveat fields every index of EA-derived data must carry.

FEEDBACK (lane/gpu, 26 Sept) fixes 1, 5 and 6: the EA "©" must survive every write and re-write of
tiles.json exactly (it was stored as mojibake after pair_gpu.py re-read the file through the Windows code
page); slope-tiles.json and visibility-tiles.json must carry the licence and attribution; the visibility
index must say that hedges, trees and buildings are not included.

    python tests/test_notices.py
"""
import inspect, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cut_tiles  # noqa: E402

EA = "\u00a9 Environment Agency copyright and/or database right 2022. All rights reserved."
OGL_URL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"


def read_index(path):
    raw = open(path, "rb").read()
    assert bytes([13]) not in raw, "CR in index"
    assert "\u00c2".encode("utf-8") not in raw and b"\\u00c2" not in raw, "mojibake in index"
    return json.loads(raw.decode("utf-8"))


def check_notice(j):
    assert j["attribution"] == EA, j["attribution"]
    assert j["licence"] == "Open Government Licence v3.0" and j["licence_url"] == OGL_URL
    assert j["derived_from"] == "Environment Agency LIDAR Composite DTM 1 m"


def test_tiles_json_keeps_the_copyright_sign_through_every_rewrite(tmp):
    """Fix 1: written by cut_tiles, then re-read and re-written by pair_gpu.write_receipt."""
    import pair_gpu
    g = 100.0 + np.zeros((257, 257))
    cut_tiles.cut_grid(g, 400000, 200000, str(tmp), "synthetic")
    j = read_index(tmp / "tiles.json")
    assert j["attribution"] == EA and cut_tiles.ATTRIBUTION == EA
    assert "\u00a9".encode("utf-8") in open(tmp / "tiles.json", "rb").read()
    assert j["licence"] == "Open Government Licence v3.0" and j["licence_url"] == OGL_URL
    pair_gpu.write_receipt(str(tmp), {"photons": 0})
    pair_gpu.write_receipt(str(tmp), {"photons": 0})            # twice: a second pass must not garble it
    j2 = read_index(tmp / "tiles.json")
    assert j2["attribution"] == EA and "receipt" in j2


def test_slope_index_has_licence_and_attribution(tmp):
    """Fix 5."""
    import slope_tiles as st
    cls = np.zeros((st.N, st.N), np.uint8); asp = np.zeros((st.N, st.N), np.uint8)
    st.write_tiles(str(tmp), cls, asp, 400000, 200000, "synthetic")
    check_notice(read_index(tmp / st.INDEX))


def test_visibility_index_has_caveat_and_attribution(tmp):
    """Fix 6."""
    import viewshed as vs
    counts = np.zeros((129, 129), np.uint8)
    vs.write_tiles(str(tmp), counts, 400000, 200000, 2.0, "synthetic", 1.7, 3.0, [(400100.0, 200100.0)])
    j = read_index(tmp / vs.INDEX)
    check_notice(j)
    assert "hedges, trees and buildings are not included" in j["caveat"]
    vs.write_tiles(str(tmp), counts, 400000, 200000, 2.0, "synthetic", 1.7, 3.0, [(400100.0, 200100.0)], canopy=True)
    assert "buildings are not included" in read_index(tmp / vs.INDEX)["caveat"]


def test_every_text_open_names_its_encoding():
    """Fix 1, at the source: no text-mode open() in src/ may fall back to the Windows code page."""
    import re
    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    bad = []
    for f in sorted(src.glob("*.py")):
        for n, line in enumerate(open(f, encoding="utf-8").read().splitlines(), 1):
            for m in re.finditer(r"\bopen\(([^()]|\([^()]*\))*\)", line):
                call = m.group(0)
                if "encoding=" not in call and not re.search(r"['\"][rwa]b['\"]", call):
                    bad.append(f"{f.name}:{n}: {call}")
    assert not bad, bad


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(*([pathlib.Path(d)] if inspect.signature(fn).parameters else []))
                print("ok  ", name)
            except Exception:
                fails += 1
                print("FAIL", name); traceback.print_exc()
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)
