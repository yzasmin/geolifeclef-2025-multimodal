# Contrat d'Interface - Étape 1 : Module Filtres → Module Statistiques

**Projet :** GeoLifeCLEF 2026 - Challenge Deep Learning MIASHS  
**Producteur :** Raihan (Module Filtres)  
**Consommateur :** Malala (Module Statistiques)  
**Fichier de référence :** [`interface_filter_to_stats.py`](interface_filter_to_stats.py)  
**Date de rédaction :** 2026-04-13

---

## Sommaire

1. [Contexte et positionnement dans le pipeline](#1-contexte-et-positionnement-dans-le-pipeline)
2. [Vue d'ensemble de l'objet de sortie](#2-vue-densemble-de-lobjet-de-sortie)
3. [Spécification détaillée des 4 clés](#3-spécification-détaillée-des-4-clés)
   - [3.1 `df_combined`](#31-df_combined--pandas-dataframe)
   - [3.2 `filter_context`](#32-filter_context--dictionnaire)
   - [3.3 `file_registry`](#33-file_registry--dictionnaire)
   - [3.4 `execution_status`](#34-execution_status--dictionnaire)
4. [Sources de données à merger](#4-sources-de-données-à-merger)
5. [Règles de merge obligatoires](#5-règles-de-merge-obligatoires)
6. [Guide d'implémentation pour Raihan](#6-guide-dimplémentation-pour-raihan)
7. [Guide de consommation pour Malala](#7-guide-de-consommation-pour-malala)
8. [Erreurs courantes et comment les éviter](#8-erreurs-courantes-et-comment-les-éviter)
9. [Exemple de sortie valide](#9-exemple-de-sortie-valide)
10. [Checklist de livraison](#10-checklist-de-livraison)

---

## 1. Contexte et positionnement dans le pipeline

Le pipeline du projet se découpe en trois étapes séquentielles :

```
┌─────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 - Raihan : Module Filtres                                  │
│                                                                     │
│  Entrées :  GLC25_PA_metadata_train.csv                             │
│             EnvironmentalValues/*.csv (BioClim, Soil, Elev, ...)    │
│             Paramètres de filtre (région, seuils, ...)              │
│                                                                     │
│  Sortie  :  FilterOutput  ◄── objet défini dans ce document         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  FilterOutput
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 - Malala : Module Statistiques                             │
│                                                                     │
│  Entrée  :  FilterOutput (ci-dessus)                                │
│  Calculs :  Statistiques descriptives par feature env               │
│             Comparaison filtre vs zone de référence                 │
│             Distribution des espèces, corrélations, etc.            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  Rapport statistique
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3 - Modèle (Guilhem / Adrian / Malala)                       │
│                                                                     │
│  Entrée  :  DataFrame enrichi + images satellites sélectionnées     │
│  Sortie  :  Prédictions d'espèces (format Kaggle CSV)               │
└─────────────────────────────────────────────────────────────────────┘
```

**Pourquoi un contrat formel ?**  
Sans interface explicite, chaque changement côté Raihan (renommage de colonne,
changement d'index, ajout d'un filtre) peut silencieusement casser le code Malala.
Ce document fixe le format de manière définitive et fournit un validateur automatique.

---

## 2. Vue d'ensemble de l'objet de sortie

La fonction de filtrage de Raihan doit **toujours** retourner un unique dictionnaire
Python de type `FilterOutput` avec exactement quatre clés :

```python
result: FilterOutput = {
    "df_combined":      pd.DataFrame,        # données fusionnées
    "filter_context":   Dict[str, Any],      # paramètres du filtre
    "file_registry":    Dict[int, str],      # chemins des fichiers images
    "execution_status": Dict[str, int|bool], # méta-info d'exécution
}
```

**Règle absolue :** aucune de ces clés ne peut être absente, même si le filtre
retourne zéro résultats. Dans ce cas, `df_combined` est un DataFrame vide
(0 ligne, colonnes présentes) et `execution_status["is_empty"]` vaut `True`.

---

## 3. Spécification détaillée des 4 clés

### 3.1 `df_combined` - Pandas DataFrame

#### Définition

Résultat d'un merge entre la table des métadonnées PA et l'ensemble des
variables environnementales, **filtré** selon les critères de Raihan.

#### Colonnes obligatoires

Ces colonnes doivent **toujours** être présentes, même si le DataFrame est vide :

| Colonne     | Type Python | Type Pandas | Description |
|-------------|-------------|-------------|-------------|
| `surveyId`  | `int`       | `int64`     | Identifiant unique du relevé terrain |
| `speciesId` | `int`       | `int64`     | Identifiant de l'espèce observée |
| `lat`       | `float`     | `float64`   | Latitude WGS84 du site |
| `lon`       | `float`     | `float64`   | Longitude WGS84 du site |

#### Colonnes environnementales attendues

Au moins une famille parmi les trois suivantes doit être présente.
L'idéal est d'inclure les trois pour que Malala ait un maximum de variables.

**BioClim - 19 variables CHELSA 1981-2010**

Source CSV : `EnvironmentalValues/ClimateAverage_1981-2010/GLC25-PA-train-bioclimatic.csv`

| Colonne | Signification |
|---------|---------------|
| `bio_1` | Température annuelle moyenne (×10, °C) |
| `bio_2` | Amplitude thermique diurne moyenne |
| `bio_3` | Isothermalité (bio_2 / bio_7 × 100) |
| `bio_4` | Saisonnalité de la température (écart-type × 100) |
| `bio_5` | Température max du mois le plus chaud |
| `bio_6` | Température min du mois le plus froid |
| `bio_7` | Amplitude thermique annuelle (bio_5 − bio_6) |
| `bio_8` | Température moyenne du trimestre le plus humide |
| `bio_9` | Température moyenne du trimestre le plus sec |
| `bio_10` | Température moyenne du trimestre le plus chaud |
| `bio_11` | Température moyenne du trimestre le plus froid |
| `bio_12` | Précipitations annuelles totales (mm) |
| `bio_13` | Précipitations du mois le plus humide |
| `bio_14` | Précipitations du mois le plus sec |
| `bio_15` | Saisonnalité des précipitations (CV) |
| `bio_16` | Précipitations du trimestre le plus humide |
| `bio_17` | Précipitations du trimestre le plus sec |
| `bio_18` | Précipitations du trimestre le plus chaud |
| `bio_19` | Précipitations du trimestre le plus froid |

> **Note :** `bio_1` et `bio_13` sont fortement corrélées avec d'autres variables.
> Raihan ne supprime **pas** ces colonnes - c'est Malala qui gère les corrélations.

**SoilGrids - 9 propriétés du sol (profondeur 5-15 cm)**

Source CSV : `EnvironmentalValues/SoilGrids/GLC25-PA-train-soilgrids.csv`

| Colonne      | Signification | Unité |
|--------------|---------------|-------|
| `soil_pH`    | pH du sol | sans unité × 10 |
| `soil_clay`  | Teneur en argile | g/kg |
| `soil_sand`  | Teneur en sable | g/kg |
| `soil_silt`  | Teneur en limon | g/kg |
| `soil_soc`   | Carbone organique du sol | dg/kg |
| `soil_nitrogen` | Azote total | cg/kg |
| `soil_cec`   | Capacité d'échange cationique | mmol(c)/kg |
| `soil_bdod`  | Densité apparente | cg/cm³ |
| `soil_cfvo`  | Fragments grossiers volumétriques | cm³/dm³ |

**Élévation**

Source CSV : `EnvironmentalValues/Elevation/GLC25-PA-train-elevation.csv`

| Colonne | Signification | Unité |
|---------|---------------|-------|
| `elev`  | Altitude du site | mètres (m) |

#### Colonnes optionnelles (à inclure si disponibles)

Ces colonnes enrichissent l'analyse Malala mais ne sont pas bloquantes :

| Colonne | Source | Description |
|---------|--------|-------------|
| `region` | Métadonnées PA | Région biogéographique (ex: `MEDITERRANEAN`) |
| `dayOfYear` | Métadonnées PA | Jour de l'année du relevé |
| `LandCover-*` | EnvironmentalValues/LandCover | Classes IGBP d'occupation du sol |
| `HumanFootprint-*` | EnvironmentalValues/HumanFootprint | 22 indicateurs OSM |

#### Contraintes sur l'index et le contenu

- **Index :** `RangeIndex` (0, 1, 2, ...) - appeler `.reset_index(drop=True)` après le merge
- **Doublons de `surveyId` :** interdits - chaque surveyId doit apparaître une seule fois
- **Valeurs manquantes :** autorisées (NaN), mais à documenter dans `filter_context`
- **Tri :** non obligatoire, mais un tri par `surveyId` est recommandé pour la lisibilité

#### Exemple minimal valide

```python
pd.DataFrame({
    "surveyId":  [1000, 1001, 1002],
    "speciesId": [42,   7,    314],
    "lat":       [43.5, 44.1, 45.8],
    "lon":       [5.2,  3.7,  6.1],
    "bio_1":     [12.3, 10.1, 8.7],   # + bio_2 ... bio_19
    "soil_pH":   [6.8,  7.1,  6.5],   # + soil_clay ... soil_cfvo
    "elev":      [523,  211,  1204],
})
```

---

### 3.2 `filter_context` - Dictionnaire

#### Définition

Dictionnaire Python libre documentant **exactement** les paramètres utilisés
pour produire `df_combined`. Il permet à Malala de contextualiser l'analyse
et de comparer les résultats filtrés à une zone de référence plus large.

#### Type

```python
FilterContext = Dict[str, Any]
# Clés : str (nom du paramètre)
# Valeurs : Any (scalaire, liste, None - pas de DataFrame ni d'objet complexe)
```

#### Clés recommandées (non exhaustives)

| Clé | Type valeur | Exemple | Description |
|-----|-------------|---------|-------------|
| `region` | `str` ou `List[str]` | `"MEDITERRANEAN"` | Région(s) biogéographique(s) filtrée(s) |
| `elevation_min` | `float` | `500.0` | Altitude minimale appliquée (m) |
| `elevation_max` | `float` | `2000.0` | Altitude maximale appliquée (m) |
| `bioclim_var` | `str` | `"bio_12"` | Variable BioClim utilisée comme critère |
| `bioclim_threshold` | `float` | `600.0` | Seuil de la variable BioClim |
| `bioclim_operator` | `str` | `">"` | Opérateur de comparaison (`>`, `<`, `==`, `between`) |
| `soil_filter` | `Dict` | `{"soil_pH": [5.5, 7.5]}` | Plages de valeurs pédologiques |
| `n_species_min` | `int` | `3` | Nombre minimum d'espèces par site |
| `date_range` | `List[int]` | `[90, 270]` | Plage de jours de l'année (dayOfYear) |
| `custom_notes` | `str` | `"Filtre test v2"` | Commentaire libre sur le filtre |

#### Contraintes

- Doit être **sérialisable en JSON** (pas de numpy arrays, pas de DataFrames)
- Doit être **non vide** : au minimum une clé décrivant le filtre appliqué
- Si aucun filtre n'est appliqué (toutes les données sont retournées),
  utiliser `{"region": "all", "filter_applied": False}`

#### Exemple

```python
filter_context = {
    "region":           "MEDITERRANEAN",
    "elevation_min":    200.0,
    "elevation_max":    1500.0,
    "bioclim_var":      "bio_12",
    "bioclim_threshold": 400.0,
    "bioclim_operator": ">",
    "n_species_min":    1,
    "custom_notes":     "Filtre zone méditerranéenne basse à moyenne altitude",
}
```

---

### 3.3 `file_registry` - Dictionnaire

#### Définition

Mapping entre chaque `surveyId` présent dans `df_combined` et le chemin
absolu vers son fichier image/série temporelle satellite.

#### Type

```python
FileRegistry = Dict[int, str]
# Clés  : int  - surveyId (doit correspondre à ceux de df_combined)
# Valeurs : str - chemin absolu vers le fichier (TIFF, .pt, .npy, ...)
#           ou None si le fichier n'existe pas pour ce surveyId
```

#### Types de fichiers référencés

| Type | Extension | Description | Répertoire source |
|------|-----------|-------------|-------------------|
| Sentinel-2 | `.tif` / `.tiff` | Patch 64×64 px, 4 bandes (R, G, B, NIR) | `SatellitePatches/` |
| Landsat TS | `.pt` | Cube PyTorch `[6, 4, 21]` (bandes × saisons × années) | `SatelliteTimeSeries-Landsat/cubes/PA-train/` |
| Bioclim TS | `.pt` | Cube PyTorch `[4, 19, 12]` (saisons × vars × mois) | `BioclimTimeSeries/cubes/PA-train/` |

> **Choix du type :** si plusieurs types sont disponibles, Raihan peut choisir
> le plus pertinent pour le filtre, ou inclure un registre par type
> (ex: `file_registry_sentinel`, `file_registry_landsat`).
> Dans ce cas, ajouter les deux clés dans `FilterOutput` et documenter ici.

#### Règle d'accès aux fichiers Sentinel-2

Le chemin TIFF suit la convention :

```
SatellitePatches/CD/AB/XXXXABCD.tiff
```

Exemples :
- `surveyId = 3018575` → `SatellitePatches/75/85/3018575.tiff`
- `surveyId = 1`       → `SatellitePatches/1/1.tiff`
- `surveyId = 42`      → `SatellitePatches/42/42.tiff`

#### Contraintes

- **Couverture :** tous les `surveyId` de `df_combined` doivent avoir une entrée
  dans `file_registry`, même si la valeur est `None`
- **Cohérence :** aucun `surveyId` dans `file_registry` ne doit être absent de `df_combined`
- **Chemins absolus :** utiliser `str(Path(...).resolve())` pour éviter les chemins relatifs
- **Fichiers manquants :** valeur `None` autorisée si le fichier n'existe pas sur le serveur

#### Exemple

```python
file_registry = {
    1000: "./data/SatellitePatches/00/10/1000.tiff",
    1001: "./data/SatellitePatches/01/10/1001.tiff",
    1002: None,   # fichier TIFF absent pour ce surveyId
}
```

---

### 3.4 `execution_status` - Dictionnaire

#### Définition

Méta-information sur l'exécution du filtre. Permet à Malala de sécuriser
ses calculs sans avoir à inspecter manuellement `df_combined`.

#### Type et clés obligatoires

```python
ExecutionStatus = {
    "count":    int,   # >= 0, nombre de lignes dans df_combined
    "is_empty": bool,  # True si et seulement si count == 0
}
```

#### Contraintes de cohérence

Ces deux règles sont vérifiées automatiquement par `build_filter_output()` :

1. `count == len(df_combined)` - le compte doit correspondre à la taille réelle du DataFrame
2. `is_empty == (count == 0)` - les deux clés doivent être cohérentes entre elles

#### Exemples

```python
# Cas normal - 342 sites trouvés
execution_status = {"count": 342, "is_empty": False}

# Cas vide - aucun site ne correspond aux critères
execution_status = {"count": 0, "is_empty": True}
```

---

## 4. Sources de données à merger

### Fichiers CSV disponibles sur le serveur

```
./data/
│
├── GLC25_PA_metadata_train.csv          ← table principale (surveyId, speciesId, lat, lon, region, ...)
├── GLC25_PA_metadata_test.csv
│
└── EnvironmentalValues/
    ├── ClimateAverage_1981-2010/
    │   └── GLC25-PA-train-bioclimatic.csv   ← bio_1 ... bio_19
    ├── SoilGrids/
    │   └── GLC25-PA-train-soilgrids.csv     ← soil_pH, soil_clay, ...
    ├── Elevation/
    │   └── GLC25-PA-train-elevation.csv     ← elev
    ├── LandCover/
    │   └── GLC25-PA-train-landcover.csv     ← LandCover-0 ... LandCover-16
    └── HumanFootprint/
        └── GLC25-PA-train-human_footprint.csv ← 22 colonnes OSM
```

### Volumétrie attendue avant filtrage

| Fichier | Lignes approx. | Clé de jointure |
|---------|---------------|-----------------|
| `GLC25_PA_metadata_train.csv` | ~500 000 | `surveyId` |
| `GLC25-PA-train-bioclimatic.csv` | ~100 000 | `surveyId` |
| `GLC25-PA-train-soilgrids.csv` | ~100 000 | `surveyId` |
| `GLC25-PA-train-elevation.csv` | ~100 000 | `surveyId` |

> Les métadonnées PA ont environ 5 lignes par `surveyId` (une par espèce observée).
> Après merge et filtrage, `df_combined` peut avoir plusieurs lignes par site.

---

## 5. Règles de merge obligatoires

### Ordre des opérations

```python
# 1. Charger la table principale (plusieurs lignes par surveyId)
meta = pd.read_csv(".../GLC25_PA_metadata_train.csv")
meta["speciesId"] = meta["speciesId"].astype(int)

# 2. Charger chaque CSV environnemental (une ligne par surveyId)
bioclim = pd.read_csv(".../GLC25-PA-train-bioclimatic.csv")
soil    = pd.read_csv(".../GLC25-PA-train-soilgrids.csv")
elev    = pd.read_csv(".../GLC25-PA-train-elevation.csv")

# Supprimer les colonnes parasites si présentes
for df in [bioclim, soil, elev]:
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

# 3. Merger les variables environnementales ensemble (sur surveyId)
env = bioclim.merge(soil, on="surveyId", how="outer") \
             .merge(elev, on="surveyId", how="outer")

# 4. Merger avec les métadonnées PA (left join pour conserver toutes les observations)
df_combined = meta.merge(env, on="surveyId", how="left")

# 5. Appliquer les filtres
df_combined = df_combined[df_combined["region"] == "MEDITERRANEAN"]
df_combined = df_combined[df_combined["elev"] >= 500]

# 6. Réinitialiser l'index
df_combined = df_combined.reset_index(drop=True)
```

### Type de join à utiliser

| Join | Quand l'utiliser |
|------|------------------|
| `how="left"` | Merge meta ← env (garder tous les relevés PA, même sans données env) |
| `how="inner"` | Merge meta ← env (garder uniquement les sites avec données env complètes) |
| `how="outer"` | Merge entre CSVs environnementaux (union de tous les surveyId) |

**Recommandation :** utiliser `how="inner"` pour le merge final si les NaN
dans les variables environnementales sont trop nombreux.

### Gestion des doublons

```python
# Supprimer les doublons si présents (même surveyId + même speciesId)
df_combined = df_combined.drop_duplicates(subset=["surveyId", "speciesId"])
```

---

## 6. Guide d'implémentation pour Raihan

### Template de fonction à implémenter

```python
from interface_filter_to_stats import build_filter_output, FilterOutput
from pathlib import Path
import pandas as pd

DATA_DIR = Path("./data")
ENV_DIR  = DATA_DIR / "EnvironmentalValues"
TIFF_DIR = DATA_DIR / "SatellitePatches"


def get_tiff_path(survey_id: int) -> str | None:
    """Construit le chemin TIFF selon la convention du challenge."""
    sid_str = str(survey_id)
    if len(sid_str) >= 4:
        subdir1 = sid_str[-2:]
        subdir2 = sid_str[-4:-2]
        path = TIFF_DIR / subdir2 / subdir1 / f"{survey_id}.tiff"
    else:
        path = TIFF_DIR / f"{survey_id}.tiff"
    return str(path) if path.exists() else None


def apply_filter(
    region: str | None = None,
    elevation_min: float = 0.0,
    elevation_max: float = 5000.0,
    bioclim_var: str | None = None,
    bioclim_min: float | None = None,
    bioclim_max: float | None = None,
) -> FilterOutput:
    """
    Applique les filtres et retourne un FilterOutput conforme au contrat.

    Parameters
    ----------
    region        : Région biogéographique à isoler (None = toutes les régions)
    elevation_min : Altitude minimale (m)
    elevation_max : Altitude maximale (m)
    bioclim_var   : Colonne BioClim à filtrer (ex: "bio_12")
    bioclim_min   : Seuil minimum pour bioclim_var
    bioclim_max   : Seuil maximum pour bioclim_var
    """
    # ── 1. Chargement ──────────────────────────────────────────────────────
    meta    = pd.read_csv(DATA_DIR / "GLC25_PA_metadata_train.csv")
    bioclim = pd.read_csv(ENV_DIR / "ClimateAverage_1981-2010/GLC25-PA-train-bioclimatic.csv")
    soil    = pd.read_csv(ENV_DIR / "SoilGrids/GLC25-PA-train-soilgrids.csv")
    elev_df = pd.read_csv(ENV_DIR / "Elevation/GLC25-PA-train-elevation.csv")

    for df in [bioclim, soil, elev_df]:
        df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

    meta["speciesId"] = meta["speciesId"].astype(int)

    # ── 2. Merge environnemental ────────────────────────────────────────────
    env = bioclim.merge(soil, on="surveyId", how="outer") \
                 .merge(elev_df, on="surveyId", how="outer")
    df  = meta.merge(env, on="surveyId", how="inner").reset_index(drop=True)

    # ── 3. Application des filtres ──────────────────────────────────────────
    context = {}

    if region is not None:
        df = df[df["region"] == region]
        context["region"] = region

    df = df[(df["elev"] >= elevation_min) & (df["elev"] <= elevation_max)]
    context["elevation_min"] = elevation_min
    context["elevation_max"] = elevation_max

    if bioclim_var is not None:
        if bioclim_min is not None:
            df = df[df[bioclim_var] >= bioclim_min]
            context[f"{bioclim_var}_min"] = bioclim_min
        if bioclim_max is not None:
            df = df[df[bioclim_var] <= bioclim_max]
            context[f"{bioclim_var}_max"] = bioclim_max

    df = df.reset_index(drop=True)

    # ── 4. Registre de fichiers ─────────────────────────────────────────────
    survey_ids = df["surveyId"].unique().tolist()
    registry   = {sid: get_tiff_path(sid) for sid in survey_ids}

    # ── 5. Retour via le constructeur validé ────────────────────────────────
    return build_filter_output(
        df_combined    = df,
        filter_context = context,
        file_registry  = registry,
    )
```

---

## 7. Guide de consommation pour Malala

### Pattern d'entrée sécurisé

```python
from interface_filter_to_stats import validate_filter_output

def compute_statistics(filter_output: dict) -> dict:
    """
    Calcule les statistiques descriptives sur un FilterOutput.
    Retourne {} si aucun site ne correspond au filtre.
    """
    # Validation stricte - lève ValueError/KeyError si le contrat est violé
    validate_filter_output(filter_output)

    # Court-circuit si résultat vide
    if filter_output["execution_status"]["is_empty"]:
        print(f"[Stats] Aucun site trouvé pour : {filter_output['filter_context']}")
        return {}

    df      = filter_output["df_combined"]
    context = filter_output["filter_context"]
    reg     = filter_output["file_registry"]
    n       = filter_output["execution_status"]["count"]

    print(f"[Stats] {n} sites filtrés - contexte : {context}")

    # Colonnes environnementales disponibles
    bio_cols  = [c for c in df.columns if c.startswith("bio_")]
    soil_cols = [c for c in df.columns if c.startswith("soil_")]
    elev_col  = "elev" if "elev" in df.columns else None

    env_cols = bio_cols + soil_cols + ([elev_col] if elev_col else [])

    stats = {
        "n_sites":          df["surveyId"].nunique(),
        "n_observations":   n,
        "n_species":        df["speciesId"].nunique(),
        "n_files_available": sum(1 for v in reg.values() if v is not None),
        "env_summary":      df[env_cols].describe().to_dict() if env_cols else {},
        "filter_context":   context,
    }

    return stats
```

### Accès typique aux données

```python
result = apply_filter(region="MEDITERRANEAN", elevation_min=500)

# Lecture directe des 4 clés
df      = result["df_combined"]       # pd.DataFrame
context = result["filter_context"]    # dict
reg     = result["file_registry"]     # dict[int, str|None]
status  = result["execution_status"]  # {"count": int, "is_empty": bool}

# Exemples d'accès
print(df[["surveyId", "lat", "lon", "bio_1", "elev"]].head())
print(f"Région filtrée : {context.get('region')}")
print(f"Fichiers disponibles : {sum(1 for v in reg.values() if v)}/{status['count']}")
```

---

## 8. Erreurs courantes et comment les éviter

| Erreur | Cause | Solution |
|--------|-------|----------|
| `KeyError: 'surveyId'` | Index non réinitialisé après merge | Appeler `.reset_index(drop=True)` |
| `ValueError: colonnes manquantes` | Colonne `bio_1` absente du CSV | Vérifier le nom exact dans le CSV source |
| `TypeError: df_combined n'est pas un DataFrame` | Passage d'un dict ou d'un numpy array | Toujours merger sur `surveyId` avec pandas |
| `ValueError: Incohérence count/is_empty` | Calcul manuel de `count` incorrect | Laisser `build_filter_output()` calculer automatiquement |
| `KeyError: 'region'` | Colonne `region` absente des métadonnées | Vérifier la colonne exacte dans `GLC25_PA_metadata_train.csv` |
| `NaN dans bio_*` | Site sans données BioClim | Normal - Malala gère les NaN ; ne pas les supprimer |
| `file_registry` incomplet | Certains `surveyId` absents du registre | Utiliser `{sid: get_tiff_path(sid) for sid in df["surveyId"].unique()}` |
| Doublons dans `df_combined` | Merge mal configuré | Appeler `.drop_duplicates(subset=["surveyId", "speciesId"])` |

---

## 9. Exemple de sortie valide

### Appel

```python
result = apply_filter(
    region        = "MEDITERRANEAN",
    elevation_min = 200.0,
    elevation_max = 1500.0,
    bioclim_var   = "bio_12",
    bioclim_min   = 400.0,
)
```

### Sortie attendue (structure)

```
FilterOutput = {

  "df_combined": pd.DataFrame (342 lignes × 33 colonnes)
    ┌──────────┬───────────┬──────┬──────┬───────┬────────┬──────────┬─────┐
    │ surveyId │ speciesId │ lat  │ lon  │ bio_1 │  ...   │ soil_pH  │elev │
    ├──────────┼───────────┼──────┼──────┼───────┼────────┼──────────┼─────┤
    │  100042  │   7821    │ 43.5 │  5.2 │  14.3 │  ...   │   6.8    │ 523 │
    │  100042  │   3317    │ 43.5 │  5.2 │  14.3 │  ...   │   6.8    │ 523 │
    │  100099  │    512    │ 44.1 │  3.7 │  13.1 │  ...   │   7.1    │ 311 │
    │   ...    │    ...    │ ...  │  ... │   ... │  ...   │   ...    │ ... │
    └──────────┴───────────┴──────┴──────┴───────┴────────┴──────────┴─────┘

  "filter_context": {
    "region":        "MEDITERRANEAN",
    "elevation_min": 200.0,
    "elevation_max": 1500.0,
    "bio_12_min":    400.0,
  }

  "file_registry": {
    100042: "./data/SatellitePatches/42/00/100042.tiff",
    100099: "./data/SatellitePatches/99/00/100099.tiff",
    100201: None,   # fichier absent pour ce surveyId
    ...             # une entrée par surveyId unique de df_combined
  }

  "execution_status": {
    "count":    342,
    "is_empty": False,
  }
}
```

---

## 10. Checklist de livraison

Avant de passer le `FilterOutput` à Malala, vérifier que :

- [ ] `build_filter_output()` s'exécute sans exception
- [ ] `validate_filter_output()` s'exécute sans exception
- [ ] `df_combined` contient les colonnes `surveyId`, `speciesId`, `lat`, `lon`
- [ ] `df_combined` contient au moins une colonne environnementale (`bio_*`, `soil_*` ou `elev`)
- [ ] L'index de `df_combined` est un `RangeIndex` (commençant à 0)
- [ ] Il n'y a pas de doublons sur `(surveyId, speciesId)`
- [ ] `filter_context` est non vide et sérialisable en JSON
- [ ] `file_registry` a une entrée pour **chaque** `surveyId` unique de `df_combined`
- [ ] `execution_status["count"]` == `len(df_combined)`
- [ ] `execution_status["is_empty"]` == `(execution_status["count"] == 0)`
- [ ] Le script `interface_filter_to_stats.py` passe son auto-test : `python interface_filter_to_stats.py`

---

*Document maintenu par Malala - toute modification de ce contrat doit être
discutée avec Raihan avant implémentation.*
