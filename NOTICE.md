# NOTICE: lidar pipeline and its output tiles

Copyright 2026 Ventus Ltd. Project: GlobalGrid2050.

**Code.** `lidar/src/*.py` and `lidar/tests/*.py` are licensed under the Apache License 2.0 (`LICENSE`).
Data is not covered by that licence (`DATA-LICENCE.md`).

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

## Copernicus DEM GLO-30 (input for `copernicus.py`, fallback outside EA coverage)
- Fetched by byte range from `https://copernicus-dem-30m.s3.amazonaws.com/` (public, no key). Access citation:
  "Copernicus Digital Elevation Model (DEM) was accessed on DATE from https://registry.opendata.aws/copernicus-dem"
  (dated in each `tiles.json`).
- Licence: the ESA User License with the Copernicus Contributing Missions annex, a free licence (short name
  "Copernicus free licence"): https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
- Adapted-data notice (verbatim; our tiles are resampled, so this is the one that applies):
  "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
  2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved".
  Citation: https://doi.org/10.5270/ESA-c5d3d65. Tiles built from it must be labelled as surface
  model, 30 m, and must not be mixed with EA tiles in the same set without both notices.

## Copernicus Sentinel-2 L2A (input for `sentinel_site.py`, `s2_fetch.py`)
- Read over HTTP range requests from Microsoft Planetary Computer, collection `sentinel-2-l2a`
  (https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a); open STAC search and anonymous SAS token, no account.
- Licence: Copernicus Sentinel data are free, full and open (Commission Delegated Regulation (EU) No 1159/2013 and
  Regulation (EU) No 377/2014; Sentinel data legal notice:
  https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
- Required credit on anything built from them (carried in every `imagery/index.json`):
  "Contains modified Copernicus Sentinel data [year]".
- Composite method: masked median (electron) against a central-rank mean of the 40th-60th percentile band
  (positron), paired on the GPU with a NumPy witness; cloud from the Sen2Cor scene classification (SCL) layer.

## Sun climate: sources checked 26 September 2026 (`sun_data.py`, `sun_cells.py`, `sun_geom.py`)

| Source | Licence and access (verbatim where quoted) | Verdict |
|---|---|---|
| **PVGIS 5.3 API**, European Commission JRC: `tmy` with `raddatabase=PVGIS-SARAH3` (satellite-derived, so cloud is included), meteo ERA5, `usehorizon=0` | "The information provided by PVGIS is free and there are no restrictions on its use." Usage conditions: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en . API rules: GET only, no AJAX from web pages, "30 calls/second per IP address" (https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en). No attribution is required; we credit it anyway. | **Used.** One call per site, sequential, named User-Agent, cached under `$WORLD_CACHE/sun/`. The page never calls it. |
| **Met Office HadUK-Grid** monthly sunshine, 1 km, 1991-2020 averages (v1.3.2.ceda, doi:10.5285/789b3065d74a4c948ab05d33556c86d0) | "Data are covered by the Open Government Licence v3.0". Access: "available to any registered CEDA user. Please Login or Register for a CEDA account" (https://catalogue.ceda.ac.uk/uuid/789b3065d74a4c948ab05d33556c86d0/). The download returned 401 without a login. | **Used** (from 26 Sept 2026). The owner downloaded `sun_hadukgrid_uk_1km_mon-30y_199101-202012.nc` (file `source` HadUK-Grid_v1.3.2.0, `version` v20260512, sha256 d1cda0d700368a29...) into `$WORLD_CACHE/sun/haduk/`. Station-based observations gridded by the Met Office. The grid mapping in the file (Airy 1830, TM origin 49N 2W, false E/N 400000/-100000, scale 0.9996012717) is checked against British National Grid; sampled bilinearly from the 4 nearest 1 km cell centres. Rechecked 26 Sept 2026: CEDA record says access "available to any registered CEDA user", licence "Open Government Licence", and "When using these data you must cite them correctly using the citation given on the CEDA Data Catalogue record"; the Met Office page (https://www.metoffice.gov.uk/hadobs/hadukgrid/) says "The HadUK-Grid datasets are freely available for use under Open Government Licence" and to "acknowledge the source if the data are used in any report or product". Web layer prefers it for sunshine hours; PVGIS stays the source for irradiance. |
| **Met Office Weather DataHub** | Own Met Office licence ("perpetual, worldwide, non-exclusive, non-transferable licence to ... publish, distribute ... exploit"), attribution "Powered by Met Office data", API key required, free plan 360 calls/day; forecasts and 48 h observations, no climate archive (https://datahub.metoffice.gov.uk/support/faqs). | **Not usable here**: no climatology, and live keyed calls are against the no-live-API rule. |
| **Met Office UK deterministic (UKV) on the AWS Open Data registry** | "British Crown copyright 2023-2025, the Met Office, is licensed under CC BY-SA"; forecast model output, rolling two-year archive (https://registry.opendata.aws/met-office-uk-deterministic/). | **Not used**: forecasts, not a climate; share-alike would bind derived tiles. |

Attribution carried in every `sun-climate.json` and `sun-cells.json`:

> Solar radiation: PVGIS 5.3 typical meteorological year, PVGIS-SARAH3 satellite radiation and ERA5 meteorology,
> European Commission Joint Research Centre. Not endorsed by the European Commission.

Carried in `sun-climate.json` under `haduk_grid` (with the CEDA citation below):

> Sunshine hours: Met Office HadUK-Grid v1.3.2.ceda, 1 km monthly averages 1991-2020, station observations gridded by
> the Met Office (doi:10.5285/789b3065d74a4c948ab05d33556c86d0). Contains public sector information licensed under the
> Open Government Licence v3.0.

Citation (verbatim from the CEDA record): Met Office; Hollis, D.; Carlisle, E.; Kendon, M.; Packman, S.; Doherty, A.
(2026): HadUK-Grid Gridded Climate Observations on a 1km grid over the UK, v1.3.2.ceda (1836-2025). NERC EDS Centre for
Environmental Data Analysis, 23 June 2026. doi:10.5285/789b3065d74a4c948ab05d33556c86d0.
Method: Hollis, D. et al. (2019) Geosci. Data J. 6(2), 151-159, doi:10.1002/gdj3.78.

Sun-cell tiles also derive from the EA LiDAR horizon, so they carry the EA OGL line as well.

Methods: Dozier, J. and Frew, J. (1990) IEEE TGRS 28(5), 963-969 (terrain shadow); NOAA Global Monitoring
Laboratory "General Solar Position Calculations" after Spencer, J.W. (1971) Search 2(5), 172; Michalsky, J.J.
(1988) Solar Energy 40(3), 227-235; Meinel, A.B. and Meinel, M.P. (1976) *Applied Solar Energy*; Laue, E.G.
(1970) Solar Energy 13(1), 43-57; Kasten, F. and Young, A.T. (1989) Applied Optics 28(22), 4735-4738;
WMO-No. 8, Guide to Instruments and Methods of Observation, Vol. I, ch. 8 (sunshine duration).

## Scripts and the data each one reads
| Script | Source data (licence) |
|---|---|
| `build_site.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `cable_geom.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `cable_sweep.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `canopy_lines.py` | EA DTM and DSMs (OGL v3.0) |
| `canopy_tiles.py` | EA DTM, First Return DSM and DSM 1 m (OGL v3.0) |
| `contour_pair.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `contour_tiles.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `contour_topo.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `copernicus.py` | Copernicus DEM GLO-30 (Copernicus free licence) |
| `copernicus_pair.py` | Copernicus DEM GLO-30 and EA DTM 1 m (both notices) |
| `cut_tiles.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `earthworks_pair.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `fetch_wcs.py` | EA LIDAR Composite DTM / First Return DSM / DSM 1 m over WCS (OGL v3.0) |
| `flow_hollows.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `flow_route.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `flow_tiles.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `geotiff_read.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `horizon_march.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `horizon_tiles.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `osgb.py` | none (Ordnance Survey method only) |
| `pair_gpu.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `piles_sweep.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `s2_fetch.py` | Copernicus Sentinel-2 L2A via Microsoft Planetary Computer (free, full and open) |
| `sentinel_site.py` | Copernicus Sentinel-2 L2A ("Contains modified Copernicus Sentinel data [year]") |
| `slope_tiles.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `sun_cells.py` | PVGIS / HadUK-Grid sun climate and EA-derived horizon tiles |
| `sun_data.py` | PVGIS 5.3 TMY (free, no restrictions) and Met Office HadUK-Grid 1 km sunshine (OGL v3.0, doi:10.5285/789b3065d74a4c948ab05d33556c86d0) |
| `sun_geom.py` | none (solar position methods only) |
| `viewshed.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |
| `viewshed_cpu.py` | EA LIDAR Composite DTM 1 m (OGL v3.0) |

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
- Flow (`flow_tiles.py`, `flow_route.py`): depression fill by Planchon, O. and Darboux, F. (2002) "A fast, simple
  and versatile algorithm to fill the depressions of digital elevation models", *Catena* 46(2-3), 159-176, witnessed
  by Barnes, R., Lehman, C. and Mulla, D. (2014) "Priority-flood: an optimal depression-filling and watershed-labeling
  algorithm for digital elevation models", *Computers & Geosciences* 62, 117-127; D8 after O'Callaghan, J.F. and
  Mark, D.M. (1984) "The extraction of drainage networks from digital elevation data", *Computer Vision, Graphics,
  and Image Processing* 28(3), 323-344; D-infinity by Tarboton, D.G. (1997) "A new method for the determination of
  flow directions and upslope areas in grid digital elevation models", *Water Resources Research* 33(2), 309-319;
  accumulation by topological peeling after Kahn, A.B. (1962) "Topological sorting of large networks",
  *Communications of the ACM* 5(11), 558-562. Output `.gfl` tiles and `flow-tiles.json` derive from the EA DTM (OGL).
- Georeferencing and grid: Ordnance Survey, *A Guide to Coordinate Systems in Great Britain* v3.6
  (2020), cited as method.

## Third-party software (installed, not vendored)
| Package | Licence | Copyright |
|---|---|---|
| NumPy | BSD-3-Clause | 2005-2025 NumPy Developers |
| tifffile | BSD-3-Clause | 2008-2026 Christoph Gohlke |
| imagecodecs | BSD-3-Clause | 2008-2026 Christoph Gohlke |
| pyproj (sun_data.py) | MIT | 2006-2026 pyproj contributors |
| netCDF4 (sun_data.py, HadUK-Grid) | MIT | 2008 Jeffrey Whitaker and netcdf4-python contributors |
| Pillow (sentinel_site.py, WebP and PNG) | MIT-CMU (HPND) | 1997-2011 Secret Labs AB, 1995-2011 Fredrik Lundh, 2010 Jeffrey A. Clark and contributors |
| CuPy (optional GPU pair) | MIT | 2015 Preferred Infrastructure, Inc.; Preferred Networks, Inc. |
| pytest (tests only, `test_copernicus.py`) | MIT | 2004 Holger Krekel and others |

If any of these is vendored or bundled, copy its LICENSE file next to it.

## Origination
The use of EA LiDAR and Copernicus DEM was first suggested in a conversation with Google Gemini. This is
recorded for history only; it is not credited on screen, by the owner's decision.

Full register for the world viewer:
`graphics-engines-open-source-world/web/world/ATTRIBUTION.md`.
