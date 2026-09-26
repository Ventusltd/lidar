# Data licence

The code in this repository is licensed under the Apache License 2.0 (see `LICENSE`).
That licence does **not** cover data.

## Terrain tiles and other data derived from Environment Agency LiDAR

Height tiles (`.ght`), slope tiles (`.gst`), their `tiles.json` indexes and any other data derived from the
Environment Agency LIDAR Composite DTM 1 m are licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Wherever they are shown or redistributed, carry this attribution:

> © Environment Agency copyright and/or database right 2022. All rights reserved.

Source: Environment Agency, LIDAR Composite Digital Terrain Model 1 m, via the Defra Data Services Platform WCS.

## Sun climate and sun-cell tiles

`sun/sun-climate.json` and `sun/tmy-hourly.bin` are derived from PVGIS (European Commission, Joint Research Centre),
whose usage conditions say "The information provided by PVGIS is free and there are no restrictions on its use."
Credit it as: Solar radiation: PVGIS 5.3 typical meteorological year, PVGIS-SARAH3 satellite radiation and ERA5
meteorology, European Commission Joint Research Centre. Not endorsed by the European Commission.

`sun/sun-climate.json` also carries `haduk_grid`: monthly sunshine hours sampled from the Met Office HadUK-Grid
v1.3.2.ceda 1 km 1991-2020 averages (station observations gridded by the Met Office), Open Government Licence v3.0,
doi:10.5285/789b3065d74a4c948ab05d33556c86d0. Credit it as: Sunshine hours: Met Office HadUK-Grid v1.3.2.ceda, 1 km
monthly averages 1991-2020, station observations gridded by the Met Office (doi:10.5285/789b3065d74a4c948ab05d33556c86d0).
Contains public sector information licensed under the Open Government Licence v3.0.

`sun/cells/*.gsc` and `sun-cells.json` combine that climate with the EA LiDAR horizon, so they are published under
OGL v3.0 and carry both the PVGIS credit and the Environment Agency line above.

## Our own written material

Documentation, receipts and reports written for this repository are licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). Commercial use is allowed.

## OpenStreetMap-derived data

Any data derived from OpenStreetMap is published only in its own files under the
[Open Database Licence (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/), credited "© OpenStreetMap contributors",
and is never merged into files under another licence.

See `NOTICE.md` for every third-party source and library.
