# NOTICE: lidar pipeline and its output tiles

Copyright 2026 Ventus Ltd. Project: GlobalGrid2050.

**Code.** This covers `lidar/src/*.py` and `lidar/tests/*.py`. **The repository has no LICENSE file yet.**
Until one is added, the code is "all rights reserved" by default, even though the repository is public.
The sister repositories use MIT (graphics-engines-open-source-world) and Apache-2.0 (GridAtlas). Add
one of those before anyone is invited to reuse the code.

**Data this pipeline reads and writes keeps its own terms.** A code licence does not cover it.

## Environment Agency LIDAR Composite DTM 1m (input; all output tiles derive from it)
- Fetched by `fetch_wcs.py` from the Defra WCS
  `https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs`.
- Licence: Open Government Licence v3.0 -
  https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
- Required attribution (verbatim from the dataset page, checked 26 Sept 2026):

  > © Environment Agency copyright and/or database right 2022. All rights reserved.

  Source: https://www.data.gov.uk/dataset/01b3ee39-da3f-47b6-83da-dc98e73a461f/lidar-composite-digital-terrain-model-dtm-1m
- **Output tiles** (`.ght` height, `.slope` slope/aspect, `tiles.json`, `site.json`) are derived from
  this data. They may be published, including commercially, under OGL. Every published tile set must
  carry the line above and the OGL link, in its index file and wherever it is displayed. Do not imply
  that the Environment Agency endorses the work. The tiles are not covered by this repository's code
  licence.
- Fetch politely: sequential requests with a named User-Agent, as the code does now. OGL does not
  guarantee the service will stay available.

## Environment Agency LIDAR Composite First Return DSM 1m and DSM (last return) 1m (canopy input)
- Fetched by `fetch_wcs.py` (products `fzdsm1m`, `lzdsm1m`) from the Defra WCS
  `https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs` and
  `https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs`.
- Licence: Open Government Licence v3.0, same attribution as the DTM (checked 26 Sept 2026 on the dataset pages):
  https://www.data.gov.uk/dataset/92534f24-0b92-4b28-9986-347cf6678b39 (first return) and
  https://www.data.gov.uk/dataset/cf3f1137-c12b-44a1-a835-e80fe4a60b92 (last return). The service lists no fees and
  no access constraints.
- The last-return composite also draws on time-series surveys, so its survey dates can differ from the first-return
  composite; `canopy_receipt.json` counts cells where last sits above first by more than 0.5 m.
- **Output**: canopy tiles (`.gcn`), `canopy-tiles.json`, `hedges.json`, under OGL with the attribution above.

## Planned inputs
- **Copernicus DEM GLO-30** (fallback outside EA coverage). Adapted-data notice (verbatim):
  "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
  2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved".
  Citation: https://doi.org/10.5270/ESA-c5d3d65. Tiles built from it must be labelled as surface
  model, 30 m, and must not be mixed with EA tiles in the same set without both notices.

## Methods
- Slope: Horn, B.K.P. (1981) "Hill shading and the reflectance map", *Proceedings of the IEEE* 69(1),
  14-47.
- Cross-check: Zevenbergen, L.W. and Thorne, C.R. (1987) "Quantitative analysis of land surface
  topography", *Earth Surface Processes and Landforms* 12(1), 47-56.
- Viewshed: Franklin, W.R. and Ray, C.K. (1994) "Higher isn't necessarily better: visibility algorithms and
  experiments", *Proc. 6th International Symposium on Spatial Data Handling*, Edinburgh, 751-770 (`viewshed.py`).
- Earth radius: Moritz, H. (2000) "Geodetic Reference System 1980", *Journal of Geodesy* 74, 128-133; refraction
  coefficient k = 0.13: Torge, W. and Mueller, J. (2012) *Geodesy*, 4th edn, de Gruyter, section 5.1.
- Contours: marching squares after Lorensen, W.E. and Cline, H.E. (1987) "Marching cubes: a high resolution 3D
  surface construction algorithm", *Computer Graphics* 21(4), 163-169; saddles after Nielson, G.M. and Hamann, B.
  (1991) "The asymptotic decider: resolving the ambiguity in marching cubes", *Proc. IEEE Visualization '91*,
  83-91; simplification by Douglas, D.H. and Peucker, T.K. (1973) "Algorithms for the reduction of the number of
  points required to represent a digitized line or its caricature", *The Canadian Cartographer* 10(2), 112-122
  (`contour_pair.py`, `contour_tiles.py`).
- Georeferencing and grid: Ordnance Survey, *A Guide to Coordinate Systems in Great Britain* v3.6
  (2020), cited as method.

## Third-party software (installed, not vendored)
| Package | Licence | Copyright |
|---|---|---|
| NumPy | BSD-3-Clause | 2005-2025 NumPy Developers |
| tifffile | BSD-3-Clause | 2008-2026 Christoph Gohlke |
| imagecodecs | BSD-3-Clause | 2008-2026 Christoph Gohlke |
| CuPy (optional GPU pair) | MIT | 2015 Preferred Infrastructure, Inc.; Preferred Networks, Inc. |

If any of these is vendored or bundled, copy its LICENSE file next to it.

## Origination
The use of EA LiDAR and Copernicus DEM was first suggested in a conversation with Google Gemini. This is
recorded for history only; it is not credited on screen, by the owner's decision.

Full register for the world viewer:
`graphics-engines-open-source-world/web/world/ATTRIBUTION.md`.
