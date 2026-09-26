# lidar

Free UK LiDAR yields 1m bare-earth 3D solar terrain.



UK ENVIRONMENT AGENCY LIDAR AND OPEN TOPOGRAPHY



LIDAR DATA OVERVIEW



LiDAR stands for Light Detection and Ranging. It is an airborne mapping technology that uses aircraft-mounted laser systems to measure ground elevations across the landscape. The sensor fires hundreds of thousands of light pulses per second to the ground and records the precise return time of each pulse.



The UK Environment Agency captures LiDAR elevation data across England to support flood risk management, coastal defense, and civil infrastructure planning.



DATASET TYPES



1\. Digital Terrain Model (DTM) - Bare Earth

The DTM processes the raw laser point cloud to filter out vegetation, trees, hedges, and surface structures. The output is a bare-earth elevation grid.

Usage for Solar: Allows direct computation of natural site topography, ground slope variations, drainage paths, structural post placement, and exact underground trench depths (0.6 meters for DC cabling and 0.9 meters for AC cabling).

2\. Digital Surface Model (DSM) - Surface Objects

The DSM preserves the top return of all physical features, including building roofs, tree canopies, and tall vegetation.

Usage for Solar: Used for 3D shading simulations, setback boundary checks, and vegetation clearance analysis.

3\. Point Cloud Data

Raw classified 3D vector point clouds (LAZ format) containing discrete laser returns classified into ground, low vegetation, medium vegetation, high vegetation, and structural points.



AVAILABLE RESOLUTIONS



Data is supplied as GeoTIFF raster tiles formatted to the Ordnance Survey national grid reference system:



\* 1 meter spatial resolution (National LIDAR Programme coverage across 99% of England)

\* 50 centimeter spatial resolution (Select coastal and high-risk river catchments)

\* 2 meter spatial resolution (Historical composite baseline archives)



LICENSING TERMS



The dataset is released under the Open Government Licence v3.0 (OGL v3.0).



Key Permitted Uses:



\* Commercial and non-commercial development without royalty fees.

\* Processing, converting, and embedding raw elevation rasters directly into custom WebGL engines, terrain meshes, or local databases.

\* Sub-licensing, publishing, or bundling derivative works with open-source software releases (compatible with Apache 2.0 and MIT licenses).



Licensing Requirements:

You must acknowledge the source of the data by including the official attribution statement in product documentation or application footers:



Attribution Text:

Contains Environment Agency information copyright Environment Agency and/or database right.



DIRECT RESOURCE LINKS



Environment Agency Data Services Portal

\[https://environment.data.gov.uk/](https://environment.data.gov.uk/?utm\_source=gemini)



National LIDAR Programme Open Data Directory

\[https://environment.data.gov.uk/dataset/2e8d0733-4f43-48b4-9e51-631c25d1b0a9](https://environment.data.gov.uk/dataset/2e8d0733-4f43-48b4-9e51-631c25d1b0a9?utm\_source=gemini)



LIDAR Survey Interactive Tile Download Map

\[https://environment.data.gov.uk/survey](https://environment.data.gov.uk/survey?utm\_source=gemini)



Defra Data Download Portal

\[https://environment.data.gov.uk/DefraDataDownload/?Mode=survey](https://environment.data.gov.uk/DefraDataDownload/?Mode=survey\&utm\_source=gemini)



Open Government Licence v3.0 Full Text

\[https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/?utm\_source=gemini)

