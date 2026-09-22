# Challenge Deep Learning MIASHS 2026

## Location-based species presence prediction

Ce projet est dedie au challenge de prediction de presence d'especes vegetales a partir de donnees geolocalisees et environnementales.

## Competition description

Le challenge consiste a predire les especes vegetales presentes a un endroit donne.
Pour chaque point geographique (GPS), les participants exploitent plusieurs types de predicteurs:

- images satellites
- series temporelles climatiques
- occupation des sols
- empreinte humaine
- autres variables environnementales

Les donnees d'observation fournies incluent environ:

- 5 millions d'occurrences Presence-Only (PO)
- 90 a 100 mille releves Presence-Absence (PA)

## Motivation

Predire les especes presentes localement est utile pour:

- produire des cartes fines de biodiversite
- estimer des indicateurs (diversite, especes menacees, invasives)
- ameliorer les outils d'identification d'especes (ex: Pl@ntNet)
- faciliter les inventaires biodiversite via des recommandations basees sur la localisation
- accelerer l'annotation et la validation d'observations citoyennes

En ecologie scientifique, cette problematique est connue sous le nom de **Species Distribution Modelling**.

## Format de soumission (Kaggle)

Pour chaque `id` du jeu de test, il faut predire un ensemble d'identifiants d'especes.

Format CSV attendu:

```csv
surveyId,predictions
1,1 52 102312
78,201 1243 1333 2310 4841
```

Contraintes:

- colonne `surveyId`: identifiants entiers des echantillons de test
- chaque `surveyId` correspond a une combinaison unique (`patchID`, `dayOfYear`)
- colonne `predictions`: liste d'identifiants `spId` separes par des espaces
- les identifiants predits doivent etre tries par ordre croissant
- aucun echantillon du test n'est vide
- les especes du test appartiennent toutes au train

## Autres ressources

En plus de la page Kaggle, consulter:

- Malpolon (framework deep learning pour SDM)
- site LifeCLEF 2025
- site FGVC12
- winning solutions / working notes:
  - GeoLifeCLEF 2022 (2nd place, B. Kellenberg et T. Devis)
  - GeoLifeCLEF 2023 (winner, H.Q. Ung et al.)
  - GeoLifeCLEF 2024 (2nd place, Yi-Chia Chen et al.)
- overview papers:
  - GeoLifeCLEF 2023
  - GeoLifeCLEF 2024
- GeoPlant Dataset

## Donnees

### Data availability

1. Toutes les donnees PA sont disponibles via Kaggle.
2. Les donnees PO sont disponibles sur le depot Seafile.
3. Les rasters originaux sont disponibles sur Seafile (ou via le dataset GeoPlant sur Kaggle).

### 1) Donnees d'observation

Le jeu d'entrainement combine:

- **Presence-Absence (PA)**: environ 100k releves, ~10k especes de la flore europeenne
- **Presence-Only (PO)**: environ 5M observations (GBIF et autres sources)

Notes importantes:

- les PA aident a corriger les faux-negatifs implicites des PO
- les PO sont echantillonnes de maniere opportuniste, donc biaises
- une absence locale en PO ne signifie pas une absence reelle

Metadonnees:

- PO: `PresenceOnlyOccurences/GLC25_PO_metadata_train.csv`
- PA: `PresenceAbsenceSurveys/GLC25_PA_metadata_train.csv`

### 2) Donnees environnementales

Pour chaque observation, des variables geographiques et environnementales sont fournies.

#### a) Satellite image patches

- patches 640m x 640m (4 bandes: R, G, B, NIR)
- taille: 64 x 64 pixels
- resolution: 10 m/pixel
- source: Sentinel-2 (pre-processing Ecodatacube)
- acces: `./SatellitePatches/` (fichiers TIFF)

Regle d'acces par `surveyId`:

- chemin: `.../CD/AB/XXXXABCD.tiff`
- exemple: `surveyId=3018575` -> `./75/85/3018575.tiff`
- cas court: `surveyId=1` -> `./1/1.tiff`

#### b) Satellite time series

Chaque observation dispose d'une serie temporelle saisonniere (depuis hiver 1999) pour 6 bandes:

- R, G, B, NIR, SWIR1, SWIR2

Formats:

- 6 CSV (un par bande), lignes = `surveyId`, colonnes = 84 saisons (hiver 2000 -> automne 2020)
- TimeSeries-Cubes (tenseurs 3D): axes `BAND`, `QUARTER`, `YEAR`

Details:

- resolution source: 30 m/pixel
- source: Landsat (Ecodatacube)
- acces: `/SatelliteTimeSeries/`

#### c) Monthly climatic rasters

- 4 variables climatiques mensuelles:
  - temperature moyenne
  - temperature min
  - temperature max
  - precipitation totale
- periode: janvier 2000 -> decembre 2019
- total: 960 rasters basse resolution (Europe)

Formats:

- CSV (un par raster, reference par `surveyId`)
- TimeSeries-Cubes (axes `RASTER_TYPE`, `YEAR`, `MONTH`)

Details:

- resolution: ~1 km
- source: CHELSA
- acces: `/EnvironmentalRasters/Climate/Climatic_Monthly_2000-2019`

#### d) Environmental rasters

Variables fournies:

- Climate
- Elevation
- Human Footprint
- LandCover
- SoilGrids

Chaque famille inclut des GeoTIFF (originaux) et des CSV de valeurs extraites.

##### Bioclimatic rasters

- 19 rasters bioclimatiques (WGS84)
- resolution: 30 arcsec (~1 km)
- source: CHELSA
- acces: `/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010`

##### Soil rasters

- 9 rasters pedologiques (proprietes du sol, profondeur 5-15 cm)
- exemples: pH, argile, carbone organique, azote (voir `definition.txt`)
- resolution: ~1 km
- source: SoilGrids
- acces: `/EnvironmentalRasters/Soilgrids`

##### Elevation

- raster haute resolution Europe
- format: GeoTIFF compresse, stockage Int16 (13.2 GB) + CSV extrait
- resolution: 1 arcsec (~30 m)
- source: ASTER GDEM V3
- acces: `/EnvironmentalRasters/Elevation`

##### Land Cover

- raster multi-bandes resolution moyenne
- classes recommandees: IGBP (17 classes) ou LCCS (43 classes)
- resolution: ~500 m
- source: MODIS Terra+Aqua 500m
- acces: `/EnvironmentalRasters/LandCover/`

##### Human footprint

- 22 rasters OSM haute resolution relies a l'empreinte humaine
- format: GeoTIFF compresse + CSV extrait
- resolution: 10-30 m
- source: Ecodatacube
- acces: `/EnvironmentalRasters/HumanFootprint/`
