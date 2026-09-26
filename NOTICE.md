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

## Planned inputs
- **Copernicus DEM GLO-30** (fallback outside EA coverage). Adapted-data notice (verbatim):
  "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
  2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved".
  Citation: https://doi.org/10.5270/ESA-c5d3d65. Tiles built from it must be labelled as surface
  model, 30 m, and must not be mixed with EA tiles in the same set without both notices.

## Sun climate: sources checked 26 September 2026 (`sun_data.py`, `sun_cells.py`)

| Source | Licence and access (verbatim where quoted) | Verdict |
|---|---|---|
| **PVGIS 5.3 API**, European Commission JRC: `tmy` with `raddatabase=PVGIS-SARAH3` (satellite-derived, so cloud is included), meteo ERA5, `usehorizon=0` | "The information provided by PVGIS is free and there are no restrictions on its use." Usage conditions: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en . API rules: GET only, no AJAX from web pages, "30 calls/second per IP address" (https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en). No attribution is required; we credit it anyway. | **Used.** One call per site, sequential, named User-Agent, cached under `E:/world-cache/sun/`. The page never calls it. |
| **Met Office HadUK-Grid** monthly sunshine, 1 km, 1991-2020 averages (v1.3.2.ceda, doi:10.5285/789b3065d74a4c948ab05d33556c86d0) | "Data are covered by the Open Government Licence v3.0". Access: "available to any registered CEDA user. Please Login or Register for a CEDA account" (https://catalogue.ceda.ac.uk/uuid/789b3065d74a4c948ab05d33556c86d0/). The download returned 401 without a login. | **Used** (from 26 Sept 2026). The owner downloaded `sun_hadukgrid_uk_1km_mon-30y_199101-202012.nc` (file `source` HadUK-Grid_v1.3.2.0, `version` v20260512, sha256 d1cda0d700368a29...) into `E:/world-cache/sun/haduk/`. Station-based observations gridded by the Met Office. The grid mapping in the file (Airy 1830, TM origin 49N 2W, false E/N 400000/-100000, scale 0.9996012717) is checked against British National Grid; sampled bilinearly from the 4 nearest 1 km cell centres. Rechecked 26 Sept 2026: CEDA record says access "available to any registered CEDA user", licence "Open Government Licence", and "When using these data you must cite them correctly using the citation given on the CEDA Data Catalogue record"; the Met Office page (https://www.metoffice.gov.uk/hadobs/hadukgrid/) says "The HadUK-Grid datasets are freely available for use under Open Government Licence" and to "acknowledge the source if the data are used in any report or product". Web layer prefers it for sunshine hours; PVGIS stays the source for irradiance. |
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

## Methods
- Slope: Horn, B.K.P. (1981) "Hill shading and the reflectance map", *Proceedings of the IEEE* 69(1),
  14-47.
- Cross-check: Zevenbergen, L.W. and Thorne, C.R. (1987) "Quantitative analysis of land surface
  topography", *Earth Surface Processes and Landforms* 12(1), 47-56.
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
| CuPy (optional GPU pair) | MIT | 2015 Preferred Infrastructure, Inc.; Preferred Networks, Inc. |

If any of these is vendored or bundled, copy its LICENSE file next to it.

## Origination
The use of EA LiDAR and Copernicus DEM was first suggested in a conversation with Google Gemini. This is
recorded for history only; it is not credited on screen, by the owner's decision.

Full register for the world viewer:
`graphics-engines-open-source-world/web/world/ATTRIBUTION.md`.
