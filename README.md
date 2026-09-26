# lidar

Free UK LiDAR turned into small, checked 3D terrain tiles for a lightweight browser site world.

## What it does

1. **Fetch** a box of the Environment Agency LIDAR Composite DTM 1 m (bare earth) from the Defra WCS 2.0.1
   service, in chunks of at most 1 km. No account or key is needed. Raw GeoTIFFs are cached outside the repo.
2. **Read** them with tifffile and imagecodecs (no GDAL), confirm British National Grid (EPSG:27700), and mask
   no-data cells.
3. **Cut** 256 m tiles at 1 m spacing (257 × 257 samples, shared edges) in the `.ght` format below, with a
   `tiles.json` index carrying a SHA-256 for every tile.
4. **Check** on the GPU with two independent formulations and CPU witnesses (`src/pair_gpu.py`): every stored
   height against the source, heights between samples two ways, and shared tile edges. Disagreements are
   counted, never suppressed, and written to a seeded receipt.
5. **Derive** slope and aspect tiles (`src/slope_tiles.py`) and earthworks volumes for trenches and platforms
   (`src/earthworks_pair.py`), each checked the same way.

```
python src/build_site.py --name open-land-01 --e 400000 --n 210000 --size 2048
python src/pair_gpu.py --tiles E:/lidar-out/open-land-01
```

## Outside EA coverage: Copernicus DEM GLO-30

`src/copernicus.py` builds `.ght` tiles from the Copernicus DEM GLO-30 where there is no EA LiDAR. It fetches
only the 1024 × 1024 blocks a box needs, by HTTP byte range from the public bucket (cached under
`E:\lidar-cache\copernicus\`), resamples to British National Grid nodes (OS Transverse Mercator + 7-parameter
Helmert, about 3.5 m, `src/osgb.py`) by bilinear interpolation, and writes tiles at **32 m** spacing (257 samples,
8,192 m a tile; never finer than 30 m, because the source holds nothing finer).

It is a **surface model** (trees and roofs included), heights are **EGM2008**, not ODN, and the stated absolute
vertical accuracy is **< 4 m LE90**. `tiles.json` records all three.

```
python src/copernicus.py --name open-land-01 --e 400128 --n 209920 --size 2048
E:/swarm/gpu-bench/venv/Scripts/python.exe src/copernicus_pair.py --ea E:/lidar-out/open-land-01 --cop E:/lidar-out/open-land-01-copernicus
```

The pair compares the tiles with the EA 1 m DTM averaged over 33 m blocks. On open-land-01 (3,969 nodes):
median −0.22 m, P25 −0.58 m, P75 +2.44 m, P95 +19.9 m, 20 % of nodes more than 5 m above bare earth (woodland and
hedges), none more than 2 m below. Open ground agrees to within a metre; tree cover does not.

## Tile format `.ght`

Little-endian. 32-byte header: magic `GGH1`, u16 version (1), u16 samples (257), u16 spacing in mm, u16 flags,
i32 south-west easting, i32 south-west northing, i32 base in cm, u16 min, u16 max, u32 no-data count.
Body: samples × samples u16, rows south to north, west to east. Height in metres = (base + q) / 100;
`0xFFFF` is no data. Heights are stored to the nearest centimetre, so the rounding error is at most 5 mm.

## Accuracy

The Environment Agency states that surveys in the composite had a vertical accuracy of ±15 cm RMSE. Heights are
metres above Ordnance Datum Newlyn. Tiles reproduce the source to within 5 mm; they cannot be more accurate than
the survey itself.

## Licences

- Code: Apache License 2.0 (`LICENSE`).
- Terrain data derived from the Environment Agency LiDAR: Open Government Licence v3.0, with the attribution
  "© Environment Agency copyright and/or database right 2022. All rights reserved." (`DATA-LICENCE.md`).
- Our own documentation and receipts: CC BY 4.0 (`DATA-LICENCE.md`).
- Every third-party source and library: `NOTICE.md`.
