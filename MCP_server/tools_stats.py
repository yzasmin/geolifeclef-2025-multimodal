"""
tools_stats.py - Module Statistiques GeoLifeCLEF 2026
======================================================

Deux interfaces d'entrée :

  1. Interface legacy (tools_data) - chargement depuis DATA_CSV_PATH
     Utilisée par tools_llm.py et les tests unitaires existants.
     Fonctions : get_species_stats, get_env_features, filter_by_features,
                 get_cooccurrences

  2. Interface FilterOutput (contrat Raihan → Malala)
     Prend en entrée le dict structuré produit par le module Filtres.
     Fonctions : analyze_filter_output, env_distribution, species_richness,
                 env_correlation, detect_outliers, bioclim_profile,
                 soil_profile, elevation_distribution, geographic_spread,
                 compare_to_reference

Toutes les fonctions retournent un dict avec la clé "ok" (bool).
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from . import tools_data
except ImportError:
    import tools_data  # type: ignore


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _error(error: str, hint: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "error": error, "hint": hint}
    payload.update(extra)
    return payload


def _detect_env_cols(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Détecte et classe les colonnes environnementales par famille."""
    cols = df.columns.tolist()
    return {
        "bioclim": [c for c in cols if c.lower().startswith("bio")],
        "soil":    [c for c in cols if c.lower().startswith("soil")],
        "elev":    [c for c in cols if c.lower() in ("elev", "elevation", "altitude")],
        "landcover":  [c for c in cols if "landcover" in c.lower() or "land_cover" in c.lower()],
        "footprint":  [c for c in cols if "footprint" in c.lower() or "human" in c.lower()],
        "climate":    [c for c in cols if c.lower() in ("temperature", "precipitation")],
    }


def _normalize_region_token(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _resolve_region_alias(region: Optional[str]) -> Optional[str]:
    if region is None:
        return None
    raw = str(region).strip()
    if not raw:
        return None

    aliases = {
        "mediterranean": "MEDITERRANEAN",
        "mediterranee": "MEDITERRANEAN",
        "alpine": "ALPINE",
        "atlantic": "ATLANTIC",
        "continental": "CONTINENTAL",
        "pannonian": "PANNONIAN",
        "boreal": "BOREAL",
        "arctic": "ARCTIC",
        "blacksea": "BLACK SEA",
        "steppic": "STEPPIC",
        "anatolian": "ANATOLIAN",
        "macaronesian": "MACARONESIAN",
    }

    key = _normalize_region_token(raw)
    if key in aliases:
        return aliases[key]

    # Cas simple: utilisateur donne déjà le code anglais avec casse variable.
    upper = raw.upper()
    if upper in set(aliases.values()):
        return upper

    return raw


def _all_env_cols(col_families: Dict[str, List[str]]) -> List[str]:
    seen, out = set(), []
    for cols in col_families.values():
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def _safe_describe(series: pd.Series) -> Dict[str, Any]:
    """Statistiques descriptives robustes sur une série numérique."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    q25, q50, q75 = s.quantile([0.25, 0.50, 0.75]).values
    return {
        "count":  int(len(s)),
        "mean":   round(float(s.mean()), 4),
        "std":    round(float(s.std()),  4),
        "min":    round(float(s.min()),  4),
        "q25":    round(float(q25),      4),
        "median": round(float(q50),      4),
        "q75":    round(float(q75),      4),
        "max":    round(float(s.max()),  4),
        "n_nan":  int(series.isna().sum()),
    }


def _validate_filter_output(data: Any) -> Optional[str]:
    """Retourne un message d'erreur ou None si valide."""
    if not isinstance(data, dict):
        return f"FilterOutput doit être un dict, reçu : {type(data).__name__}"
    required = {"df_combined", "filter_context", "file_registry", "execution_status"}
    missing = required - data.keys()
    if missing:
        return f"Clés manquantes dans FilterOutput : {missing}"
    if not isinstance(data["df_combined"], pd.DataFrame):
        return "df_combined doit être un pd.DataFrame"
    status = data["execution_status"]
    if not isinstance(status, dict) or "count" not in status or "is_empty" not in status:
        return "execution_status doit contenir 'count' et 'is_empty'"
    return None


# ===========================================================================
# INTERFACE 1 - Fonctions legacy (tools_data / DATA_CSV_PATH)
# ===========================================================================

def get_species_stats(top_k: int = 10, rare_threshold: int = 2) -> Dict[str, Any]:
    """
    Fréquences d'occurrence des espèces dans le dataset chargé.

    Parameters
    ----------
    top_k           : nombre d'espèces les plus fréquentes à retourner
    rare_threshold  : seuil d'occurrence en dessous duquel une espèce est
                      considérée rare

    Returns
    -------
    {
      "ok": True,
      "total_species": int,
      "rare_count": int,
      "top_species": [{"species_id": ..., "survey_count": int}, ...],
      "frequency_summary": {mean, std, min, median, max}
    }
    """
    df, err = tools_data._load_dataframe()
    if err:
        return err

    columns = tools_data._resolve_standard_columns(df)
    species_col = columns.get("species_id")
    survey_col  = columns.get("survey_id")

    if species_col is None:
        return _error(
            "Missing species column",
            "Expected a speciesId or spId column in the dataset.",
            detected_columns=columns,
        )

    counts = df[species_col].dropna().value_counts()
    total  = int(len(counts))
    rare   = int((counts <= rare_threshold).sum())

    top = [
        {"species_id": tools_data._safe_value(sp), "survey_count": int(c)}
        for sp, c in counts.head(max(1, int(top_k))).items()
    ]

    freq_arr = counts.values.astype(float)
    freq_summary = {
        "mean":   round(float(freq_arr.mean()), 2),
        "std":    round(float(freq_arr.std()),  2),
        "min":    int(freq_arr.min()),
        "median": round(float(np.median(freq_arr)), 2),
        "max":    int(freq_arr.max()),
    }

    result: Dict[str, Any] = {
        "ok":              True,
        "total_species":   total,
        "rare_count":      rare,
        "rare_threshold":  rare_threshold,
        "top_species":     top,
        "frequency_summary": freq_summary,
    }

    if survey_col is not None:
        result["total_surveys"] = int(df[survey_col].nunique())

    return result


def get_env_features(survey_id: Any) -> Dict[str, Any]:
    """
    Variables environnementales d'un site identifié par son surveyId.

    Returns
    -------
    {
      "ok": True,
      "survey_id": str,
      "location": {"lat": float, "lon": float, "region": str, "country": str},
      "env_features": {col: value, ...}
    }
    """
    df, err = tools_data._load_dataframe()
    if err:
        return err

    columns    = tools_data._resolve_standard_columns(df)
    survey_col = columns.get("survey_id")

    if survey_col is None:
        return _error(
            "Missing survey column",
            "Expected a surveyId column in the dataset.",
            detected_columns=columns,
        )

    sid = tools_data._normalize_survey_id(survey_id)
    if sid is None:
        return _error("Invalid survey_id", "Provide a valid survey identifier.")

    rows = df[df[survey_col].astype(str).str.replace(r"\.0$", "", regex=True) == sid]
    if rows.empty:
        return _error("Survey not found", "No row matches this survey_id.", survey_id=sid)

    row = rows.iloc[0]

    # Localisation
    location: Dict[str, Any] = {}
    for key, candidates in [
        ("lat",     ["lat", "latitude"]),
        ("lon",     ["lon", "longitude"]),
        ("region",  ["region"]),
        ("country", ["country"]),
    ]:
        for c in candidates:
            if c in df.columns:
                location[key] = tools_data._safe_value(row[c])
                break

    # Colonnes environnementales (toutes sauf ID / localisation / labels)
    skip = {survey_col, "lat", "latitude", "lon", "longitude",
            "region", "country", "speciesId", "spId", "species_id"}
    if columns.get("species_id"):
        skip.add(columns["species_id"])

    env_features: Dict[str, Any] = {}
    for col in df.columns:
        if col in skip:
            continue
        val = row[col]
        if pd.notna(val):
            env_features[col] = tools_data._safe_value(val)

    return {
        "ok":          True,
        "survey_id":   sid,
        "location":    location,
        "env_features": env_features,
    }


def filter_by_features(
    min_altitude:  Optional[float] = None,
    max_altitude:  Optional[float] = None,
    min_temperature: Optional[float] = None,
    max_temperature: Optional[float] = None,
    country:       Optional[str]   = None,
    region:        Optional[str]   = None,
    min_bio1:      Optional[float] = None,
    max_bio1:      Optional[float] = None,
    min_bio12:     Optional[float] = None,
    max_bio12:     Optional[float] = None,
    limit:         int             = 100,
) -> Dict[str, Any]:
    """
    Filtre le dataset legacy par critères environnementaux.

    Returns
    -------
    {
      "ok": True,
      "matched_count": int,
      "filters_applied": dict,
      "sample": [list of dicts, capped at limit]
    }

    Notes
    -----
    Compatibilité MCP :
      - min_temperature / max_temperature utilisent en priorité une colonne
        explicite de température (ex: temperature).
      - Sinon, fallback sur Bio1 / bio_1 avec conversion ×0.1 lorsque les
        valeurs semblent stockées au format BioClim CHELSA (dixièmes de °C).
    """
    df, err = tools_data._load_dataframe()
    if err:
        return err

    mask = pd.Series([True] * len(df), index=df.index)
    applied: Dict[str, Any] = {}
    warnings_out: List[str] = []

    # Altitude / Élévation
    elev_col = next(
        (c for c in df.columns if c.lower() in ("elevation", "elev", "altitude")),
        None,
    )
    if elev_col is not None:
        elev = pd.to_numeric(df[elev_col], errors="coerce")
        if min_altitude is not None:
            mask &= elev >= min_altitude
            applied["min_altitude"] = min_altitude
        if max_altitude is not None:
            mask &= elev <= max_altitude
            applied["max_altitude"] = max_altitude
    elif min_altitude is not None or max_altitude is not None:
        msg = "Altitude filter requested but no elevation column was found."
        applied["altitude_filter_skipped"] = msg
        warnings_out.append(msg)

    # Pays
    if country is not None and "country" in df.columns:
        mask &= df["country"].astype(str).str.lower() == country.lower()
        applied["country"] = country

    # Région
    if region is not None and "region" in df.columns:
        mask &= df["region"].astype(str).str.lower().str.contains(region.lower(), na=False)
        applied["region"] = region

    # BioClim 1 (utilisé aussi en fallback pour température)
    bio1_col = next(
        (c for c in df.columns if c.lower() in ("bio1", "bio_1")),
        None,
    )

    # Température (compat server.py: min_temperature / max_temperature)
    if min_temperature is not None or max_temperature is not None:
        temp_col = next(
            (
                c for c in df.columns
                if c.lower() in (
                    "temperature",
                    "temp",
                    "mean_temperature",
                    "avg_temperature",
                )
            ),
            None,
        )
        temperature = None
        source = None
        scaled_by_10 = False

        if temp_col is not None:
            temperature = pd.to_numeric(df[temp_col], errors="coerce")
            source = temp_col
        elif bio1_col is not None:
            temperature = pd.to_numeric(df[bio1_col], errors="coerce")
            source = f"{bio1_col} (fallback)"
            median_abs = float(temperature.abs().median(skipna=True)) if not temperature.dropna().empty else 0.0
            # CHELSA BioClim est souvent stocké en dixièmes de degrés.
            if median_abs > 80:
                temperature = temperature / 10.0
                scaled_by_10 = True

        if temperature is not None:
            if min_temperature is not None:
                mask &= temperature >= min_temperature
                applied["min_temperature"] = min_temperature
            if max_temperature is not None:
                mask &= temperature <= max_temperature
                applied["max_temperature"] = max_temperature
            applied["temperature_source"] = source
            if scaled_by_10:
                applied["temperature_scale_factor"] = 0.1
        else:
            msg = "No temperature column found; temperature filter was not applied."
            applied["temperature_filter_skipped"] = msg
            warnings_out.append(msg)

    # BioClim 1 (température annuelle moyenne)
    if bio1_col is not None:
        bio1 = pd.to_numeric(df[bio1_col], errors="coerce")
        if min_bio1 is not None:
            mask &= bio1 >= min_bio1
            applied["min_bio1"] = min_bio1
        if max_bio1 is not None:
            mask &= bio1 <= max_bio1
            applied["max_bio1"] = max_bio1

    # BioClim 12 (précipitations annuelles)
    bio12_col = next(
        (c for c in df.columns if c.lower() in ("bio12", "bio_12")),
        None,
    )
    if bio12_col is not None:
        bio12 = pd.to_numeric(df[bio12_col], errors="coerce")
        if min_bio12 is not None:
            mask &= bio12 >= min_bio12
            applied["min_bio12"] = min_bio12
        if max_bio12 is not None:
            mask &= bio12 <= max_bio12
            applied["max_bio12"] = max_bio12

    filtered = df[mask].head(int(limit))

    return {
        "ok":             True,
        "matched_count":  int(mask.sum()),
        "filters_applied": applied,
        "warnings":       warnings_out,
        "sample":         filtered.to_dict(orient="records"),
    }


def get_cooccurrences(species_id: Any, top_k: int = 10) -> Dict[str, Any]:
    """
    Espèces co-occurrant le plus fréquemment avec l'espèce cible.

    Parameters
    ----------
    species_id : identifiant de l'espèce cible
    top_k      : nombre de co-espèces à retourner

    Returns
    -------
    {
      "ok": True,
      "target_species_id": ...,
      "target_survey_count": int,
      "cooccurrences": [{"species_id": ..., "shared_surveys": int, "jaccard": float}, ...]
    }
    """
    df, err = tools_data._load_dataframe()
    if err:
        return err

    columns    = tools_data._resolve_standard_columns(df)
    survey_col = columns.get("survey_id")
    species_col = columns.get("species_id")

    if survey_col is None or species_col is None:
        return _error(
            "Missing columns",
            "Expected surveyId and speciesId columns.",
            detected_columns=columns,
        )

    # Normalisation des identifiants pour comparaison robuste
    target = tools_data._safe_value(species_id)
    target_norm = str(target).strip()
    if target_norm.endswith(".0") and target_norm[:-2].isdigit():
        target_norm = target_norm[:-2]

    valid_mask = df[survey_col].notna() & df[species_col].notna()
    if not bool(valid_mask.any()):
        return _error(
            "No valid survey/species rows",
            "Dataset has no non-null survey/species pairs.",
        )

    pairs = pd.DataFrame(
        {
            "survey_norm": (
                df.loc[valid_mask, survey_col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            ),
            "species_norm": (
                df.loc[valid_mask, species_col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            ),
            "species_raw": df.loc[valid_mask, species_col],
        }
    ).drop_duplicates(subset=["survey_norm", "species_norm"])

    # Surveys contenant l'espèce cible
    target_mask = pairs["species_norm"] == target_norm
    target_surveys = set(pairs.loc[target_mask, "survey_norm"].tolist())
    n_target = len(target_surveys)

    if n_target == 0:
        return _error(
            "Species not found",
            "No survey contains this species_id.",
            species_id=target,
        )

    # Comptes partagés: n° de surveys du target contenant aussi chaque espèce
    co_pairs = pairs[
        pairs["survey_norm"].isin(target_surveys) &
        (pairs["species_norm"] != target_norm)
    ]
    shared_counts = co_pairs.groupby("species_norm")["survey_norm"].nunique()

    # Comptes totaux par espèce sur tout le dataset
    total_counts = pairs.groupby("species_norm")["survey_norm"].nunique()

    # Représentant "raw" pour conserver un species_id lisible
    raw_species = (
        pairs.drop_duplicates(subset=["species_norm"])
        .set_index("species_norm")["species_raw"]
    )

    ranked = pd.DataFrame({"shared_surveys": shared_counts}).join(
        total_counts.rename("total_surveys"),
        how="left",
    )
    ranked["union_surveys"] = ranked["total_surveys"] + n_target - ranked["shared_surveys"]
    ranked["jaccard"] = (ranked["shared_surveys"] / ranked["union_surveys"]).fillna(0.0).round(4)
    ranked = ranked.sort_values(["shared_surveys", "jaccard"], ascending=[False, False])

    cooccurrences: List[Dict[str, Any]] = []
    for sp_norm, row in ranked.head(max(1, int(top_k))).iterrows():
        raw_val = raw_species.get(sp_norm, sp_norm)
        cooccurrences.append(
            {
                "species_id": tools_data._safe_value(raw_val),
                "shared_surveys": int(row["shared_surveys"]),
                "jaccard": float(row["jaccard"]),
            }
        )

    return {
        "ok":                True,
        "target_species_id": target,
        "target_survey_count": n_target,
        "cooccurrences":     cooccurrences,
    }


def get_top_drivers(
    region: Optional[str] = None,
    country: Optional[str] = None,
    elevation_min: float = 0.0,
    elevation_max: float = 9000.0,
    bioclim_var: Optional[str] = None,
    bioclim_min: Optional[float] = None,
    bioclim_max: Optional[float] = None,
    n_species_min: Optional[int] = None,
    top_k_species: int = 10,
    top_k_vars: int = 8,
) -> Dict[str, Any]:
    """
    Interface MCP compacte pour récupérer les variables environnementales
    les plus influentes sur une zone filtrée.

    Cette fonction bridge l'interface Raihan (apply_filter) vers l'analyse
    Malala (top_drivers) et renvoie un JSON prêt pour l'agent.
    """
    resolved_region = _resolve_region_alias(region)

    try:
        filter_output = tools_data.apply_filter(
            region=resolved_region,
            country=country,
            elevation_min=float(elevation_min),
            elevation_max=float(elevation_max),
            bioclim_var=bioclim_var,
            bioclim_min=None if bioclim_min is None else float(bioclim_min),
            bioclim_max=None if bioclim_max is None else float(bioclim_max),
            n_species_min=None if n_species_min is None else int(n_species_min),
        )
    except Exception as exc:
        return _error(
            "Failed to apply filters",
            "Could not build FilterOutput from tools_data.apply_filter.",
            details=str(exc),
        )

    status = filter_output.get("execution_status", {})
    if bool(status.get("is_empty", False)):
        return {
            "ok": True,
            "filter_context": filter_output.get("filter_context", {}),
            "execution_status": status,
            "n_sites": 0,
            "n_env_vars": 0,
            "drivers": [],
            "family_importance": {},
            "llm_summary": "Aucun site ne correspond au filtre demandé.",
        }

    result = top_drivers(
        filter_output=filter_output,
        top_k_species=max(1, int(top_k_species)),
        top_k_vars=max(1, int(top_k_vars)),
    )
    if not result.get("ok"):
        return result

    result["filter_context"] = filter_output.get("filter_context", {})
    result["execution_status"] = status
    result["analysis_source"] = "tools_stats.top_drivers"
    if region is not None:
        result["region_input"] = region
        result["region_resolved"] = resolved_region
    return result


# ===========================================================================
# INTERFACE 2 - Analyses statistiques sur FilterOutput (contrat Raihan → Malala)
# ===========================================================================

def analyze_filter_output(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Point d'entrée principal de Malala : calcule toutes les analyses
    statistiques sur un FilterOutput produit par Raihan.

    Returns
    -------
    {
      "ok": True,
      "filter_context": dict,
      "execution_status": dict,
      "n_sites": int,
      "n_observations": int,
      "n_species": int,
      "n_files_available": int,
      "env_distribution": dict,       ← _safe_describe par colonne env
      "species_richness": dict,       ← stats de richesse par site
      "geographic_spread": dict,      ← bbox, centroid, spread
      "bioclim_profile": dict,        ← stats BioClim
      "soil_profile": dict,           ← stats SoilGrids
      "elevation_distribution": dict, ← stats élévation
      "outlier_sites": list,          ← surveyId avec z-score élevé
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    status = filter_output["execution_status"]
    if status["is_empty"]:
        return {
            "ok":              True,
            "filter_context":  filter_output["filter_context"],
            "execution_status": status,
            "n_sites":          0,
            "n_observations":   0,
            "n_species":        0,
            "n_files_available": 0,
            "warning":         "Aucun site ne correspond aux critères de filtrage.",
        }

    df  = filter_output["df_combined"]
    reg = filter_output["file_registry"]

    n_sites   = int(df["surveyId"].nunique()) if "surveyId" in df.columns else int(len(df))
    n_obs     = int(len(df))
    n_species = int(df["speciesId"].nunique()) if "speciesId" in df.columns else 0
    n_files   = sum(1 for v in reg.values() if v is not None)

    return {
        "ok":               True,
        "filter_context":   filter_output["filter_context"],
        "execution_status": status,
        "n_sites":          n_sites,
        "n_observations":   n_obs,
        "n_species":        n_species,
        "n_files_available": n_files,
        "env_distribution":       env_distribution(filter_output),
        "species_richness":       species_richness(filter_output),
        "geographic_spread":      geographic_spread(filter_output),
        "bioclim_profile":        bioclim_profile(filter_output),
        "soil_profile":           soil_profile(filter_output),
        "elevation_distribution": elevation_distribution(filter_output),
        "outlier_sites":          detect_outliers(filter_output)["outlier_sites"],
        "species_composition":    species_composition(filter_output),
        "diversity_indices":      diversity_indices(filter_output),
        "species_env_correlation": species_env_correlation(filter_output),
        "top_drivers":            top_drivers(filter_output),
    }


def env_distribution(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Statistiques descriptives (count, mean, std, min, q25, median, q75, max, n_nan)
    pour chaque variable environnementale présente dans df_combined.

    Returns
    -------
    {
      "ok": True,
      "n_env_cols": int,
      "columns": {col_name: {count, mean, std, min, q25, median, q75, max, n_nan}, ...}
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_env_cols": 0, "columns": {}}

    df         = filter_output["df_combined"]
    col_fams   = _detect_env_cols(df)
    env_cols   = _all_env_cols(col_fams)

    stats: Dict[str, Any] = {}
    for col in env_cols:
        if col in df.columns:
            stats[col] = _safe_describe(df[col])

    return {
        "ok":        True,
        "n_env_cols": len(stats),
        "families":  {fam: cols for fam, cols in col_fams.items() if cols},
        "columns":   stats,
    }


def species_richness(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Richesse spécifique (nombre d'espèces distinctes) par site (surveyId).

    Returns
    -------
    {
      "ok": True,
      "n_sites": int,
      "richness_per_site": [{"surveyId": int, "n_species": int, "lat": float, "lon": float}, ...],
      "richness_summary": {mean, std, min, median, max},
      "top_rich_sites": [...],   ← top 10 sites les plus riches
      "top_poor_sites": [...],   ← top 10 sites les moins riches
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_sites": 0, "richness_per_site": [], "richness_summary": {}}

    df = filter_output["df_combined"]

    if "surveyId" not in df.columns or "speciesId" not in df.columns:
        return _error(
            "Missing columns",
            "df_combined doit contenir surveyId et speciesId pour calculer la richesse.",
        )

    agg: Dict[str, Any] = {"n_species": ("speciesId", "nunique")}
    if "lat" in df.columns:
        agg["lat"] = ("lat", "first")
    if "lon" in df.columns:
        agg["lon"] = ("lon", "first")

    richness_df = df.groupby("surveyId").agg(**agg).reset_index()
    richness_arr = richness_df["n_species"].values.astype(float)

    q25, q50, q75 = np.percentile(richness_arr, [25, 50, 75])
    summary = {
        "mean":   round(float(richness_arr.mean()), 2),
        "std":    round(float(richness_arr.std()),  2),
        "min":    int(richness_arr.min()),
        "q25":    round(float(q25), 2),
        "median": round(float(q50), 2),
        "q75":    round(float(q75), 2),
        "max":    int(richness_arr.max()),
    }

    records = richness_df.to_dict(orient="records")

    sorted_desc = richness_df.sort_values("n_species", ascending=False)
    sorted_asc  = richness_df.sort_values("n_species", ascending=True)

    return {
        "ok":                True,
        "n_sites":           int(len(richness_df)),
        "richness_per_site": records,
        "richness_summary":  summary,
        "top_rich_sites":    sorted_desc.head(10).to_dict(orient="records"),
        "top_poor_sites":    sorted_asc.head(10).to_dict(orient="records"),
    }


def env_correlation(
    filter_output: Dict[str, Any],
    method: str = "pearson",
    min_obs: int = 5,
) -> Dict[str, Any]:
    """
    Matrice de corrélation entre les variables environnementales.

    Parameters
    ----------
    method  : "pearson" | "spearman" | "kendall"
    min_obs : nombre minimum d'observations non-NaN requis par paire

    Returns
    -------
    {
      "ok": True,
      "method": str,
      "n_obs": int,
      "columns": [str, ...],
      "matrix": [[float, ...], ...],   ← corrélation arrondie à 3 décimales
      "top_pairs": [{"col_a", "col_b", "r"}, ...]  ← top 15 corrélations |r| > 0.7
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "method": method, "n_obs": 0, "columns": [], "matrix": [], "top_pairs": []}

    df       = filter_output["df_combined"]
    col_fams = _detect_env_cols(df)
    env_cols = [c for c in _all_env_cols(col_fams) if c in df.columns]

    # Garder uniquement les colonnes numériques avec assez d'observations
    num_cols = []
    for c in env_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= min_obs:
            num_cols.append(c)

    if len(num_cols) < 2:
        return _error(
            "Not enough numeric columns",
            f"Au moins 2 colonnes numériques avec ≥ {min_obs} observations requises.",
            available=num_cols,
        )

    num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
    corr   = num_df.corr(method=method).round(3)

    matrix = corr.values.tolist()

    # Paires fortement corrélées (|r| > 0.7, hors diagonale)
    top_pairs = []
    for i, ca in enumerate(num_cols):
        for j, cb in enumerate(num_cols):
            if j <= i:
                continue
            r = corr.loc[ca, cb]
            if not math.isnan(r) and abs(r) > 0.7:
                top_pairs.append({"col_a": ca, "col_b": cb, "r": round(float(r), 3)})

    top_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

    return {
        "ok":      True,
        "method":  method,
        "n_obs":   int(len(num_df.dropna())),
        "columns": num_cols,
        "matrix":  matrix,
        "top_pairs": top_pairs[:15],
    }


def detect_outliers(
    filter_output: Dict[str, Any],
    z_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Identifie les sites dont au moins une variable environnementale
    présente un z-score supérieur à z_threshold.

    Returns
    -------
    {
      "ok": True,
      "z_threshold": float,
      "n_outliers": int,
      "outlier_sites": [
        {"surveyId": int, "outlier_cols": [{"col", "value", "z_score"}]}, ...
      ]
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "z_threshold": z_threshold, "n_outliers": 0, "outlier_sites": []}

    df       = filter_output["df_combined"]
    col_fams = _detect_env_cols(df)
    env_cols = [c for c in _all_env_cols(col_fams) if c in df.columns]

    survey_col = "surveyId" if "surveyId" in df.columns else None

    # Agréger par site (une ligne par surveyId) pour ne pas compter en double
    if survey_col:
        site_df = df.drop_duplicates(subset=[survey_col]).set_index(survey_col)
    else:
        site_df = df.copy()

    num_env = [c for c in env_cols if c in site_df.columns]

    # Calcul vectorisé : z-score colonne par colonne sur site_df entier
    num_df = site_df[num_env].apply(pd.to_numeric, errors="coerce")
    mu     = num_df.mean()
    sigma  = num_df.std().replace(0, float("nan"))
    z_df   = (num_df - mu).abs() / sigma          # DataFrame de z-scores

    # Masque booléen : True si z >= seuil
    flagged_mask = z_df >= z_threshold             # même shape que z_df

    outlier_sites: List[Dict[str, Any]] = []
    # Itérer uniquement sur les sites qui ont au moins un outlier
    outlier_idx = flagged_mask.any(axis=1)
    for sid in site_df.index[outlier_idx]:
        row_z   = z_df.loc[sid]
        row_val = num_df.loc[sid]
        cols_flagged = [
            {
                "col":     col,
                "value":   round(float(row_val[col]), 4),
                "z_score": round(float(row_z[col]),   3),
            }
            for col in num_env
            if flagged_mask.loc[sid, col] and not pd.isna(row_val[col])
        ]
        if cols_flagged:
            outlier_sites.append({
                "surveyId":    sid,
                "outlier_cols": sorted(cols_flagged, key=lambda x: x["z_score"], reverse=True),
            })

    return {
        "ok":           True,
        "z_threshold":  z_threshold,
        "n_outliers":   len(outlier_sites),
        "outlier_sites": outlier_sites,
    }


def bioclim_profile(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profil statistique des 19 variables BioClim CHELSA dans le jeu filtré.

    Returns
    -------
    {
      "ok": True,
      "n_bioclim_cols": int,
      "variables": {
        "bio_1": {count, mean, std, min, q25, median, q75, max, n_nan},
        ...
      },
      "temperature_vars": [cols bio_1..bio_11],
      "precipitation_vars": [cols bio_12..bio_19],
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_bioclim_cols": 0, "variables": {}}

    df      = filter_output["df_combined"]
    bio_cols = _detect_env_cols(df)["bioclim"]

    if not bio_cols:
        return {
            "ok":            True,
            "n_bioclim_cols": 0,
            "variables":     {},
            "warning":       "Aucune colonne BioClim détectée dans df_combined.",
        }

    variables = {col: _safe_describe(df[col]) for col in bio_cols if col in df.columns}

    # Classification température / précipitations (convention CHELSA bio_1..11 / bio_12..19)
    temp_vars  = [c for c in bio_cols if any(c.endswith(str(i)) for i in range(1, 12))]
    precip_vars = [c for c in bio_cols if any(c.endswith(str(i)) for i in range(12, 20))]

    return {
        "ok":                True,
        "n_bioclim_cols":    len(variables),
        "variables":         variables,
        "temperature_vars":  temp_vars,
        "precipitation_vars": precip_vars,
    }


def soil_profile(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profil statistique des variables pédologiques SoilGrids.

    Returns
    -------
    {
      "ok": True,
      "n_soil_cols": int,
      "variables": {col: {count, mean, std, min, q25, median, q75, max, n_nan}, ...}
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_soil_cols": 0, "variables": {}}

    df        = filter_output["df_combined"]
    soil_cols = _detect_env_cols(df)["soil"]

    if not soil_cols:
        return {
            "ok":          True,
            "n_soil_cols": 0,
            "variables":   {},
            "warning":     "Aucune colonne SoilGrids détectée dans df_combined.",
        }

    variables = {col: _safe_describe(df[col]) for col in soil_cols if col in df.columns}

    return {
        "ok":          True,
        "n_soil_cols": len(variables),
        "variables":   variables,
    }


def elevation_distribution(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Distribution de l'élévation avec histogramme par tranches de 200 m.

    Returns
    -------
    {
      "ok": True,
      "col_used": str,
      "summary": {count, mean, std, min, q25, median, q75, max},
      "histogram": [{"bin_label": "0-200m", "count": int, "pct": float}, ...]
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "col_used": None, "summary": {}, "histogram": []}

    df       = filter_output["df_combined"]
    elev_col = next(
        (c for c in _detect_env_cols(df)["elev"] if c in df.columns),
        None,
    )

    if elev_col is None:
        return {
            "ok":      True,
            "col_used": None,
            "summary": {},
            "histogram": [],
            "warning": "Aucune colonne d'élévation détectée dans df_combined.",
        }

    # Dédupliquer sur surveyId pour ne pas compter une altitude plusieurs fois
    if "surveyId" in df.columns:
        elev_series = df.drop_duplicates(subset=["surveyId"])[elev_col]
    else:
        elev_series = df[elev_col]

    elev_num = pd.to_numeric(elev_series, errors="coerce").dropna()

    summary = _safe_describe(elev_series)

    # Histogramme par tranches de 200 m
    e_min = math.floor(float(elev_num.min()) / 200) * 200
    e_max = math.ceil(float(elev_num.max()) / 200) * 200 + 200
    bins  = list(range(int(e_min), int(e_max), 200))

    histogram = []
    total_sites = len(elev_num)
    for lo in bins[:-1]:
        hi    = lo + 200
        count = int(((elev_num >= lo) & (elev_num < hi)).sum())
        histogram.append({
            "bin_label": f"{lo}-{hi}m",
            "count":     count,
            "pct":       round(100 * count / total_sites, 1) if total_sites > 0 else 0.0,
        })

    return {
        "ok":       True,
        "col_used": elev_col,
        "summary":  summary,
        "histogram": histogram,
    }


def geographic_spread(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emprise géographique et répartition spatiale des sites filtrés.

    Returns
    -------
    {
      "ok": True,
      "n_sites": int,
      "centroid": {"lat": float, "lon": float},
      "bbox": {"lat_min", "lat_max", "lon_min", "lon_max"},
      "lat_spread_km": float,   ← distance N-S approx.
      "lon_spread_km": float,   ← distance E-O approx.
      "lat_std": float,
      "lon_std": float,
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_sites": 0, "centroid": {}, "bbox": {}}

    df = filter_output["df_combined"]

    if "lat" not in df.columns or "lon" not in df.columns:
        return _error(
            "Missing coordinates",
            "df_combined doit contenir les colonnes lat et lon.",
        )

    # Dédupliquer sur surveyId pour avoir une coordonnée par site
    if "surveyId" in df.columns:
        sites = df.drop_duplicates(subset=["surveyId"])[["surveyId", "lat", "lon"]]
    else:
        sites = df[["lat", "lon"]].drop_duplicates()

    lat = pd.to_numeric(sites["lat"], errors="coerce").dropna()
    lon = pd.to_numeric(sites["lon"], errors="coerce").dropna()

    if lat.empty or lon.empty:
        return _error("No valid coordinates", "Les colonnes lat/lon ne contiennent que des NaN.")

    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    centroid_lat     = float(lat.mean())
    centroid_lon     = float(lon.mean())

    # Conversion approximative degrés → km (1° lat ≈ 111 km)
    lat_km = round((lat_max - lat_min) * 111.0, 1)
    lon_km = round((lon_max - lon_min) * 111.0 * math.cos(math.radians(centroid_lat)), 1)

    return {
        "ok":      True,
        "n_sites": int(len(sites)),
        "centroid": {
            "lat": round(centroid_lat, 5),
            "lon": round(centroid_lon, 5),
        },
        "bbox": {
            "lat_min": round(lat_min, 5),
            "lat_max": round(lat_max, 5),
            "lon_min": round(lon_min, 5),
            "lon_max": round(lon_max, 5),
        },
        "lat_spread_km": lat_km,
        "lon_spread_km": lon_km,
        "lat_std": round(float(lat.std()), 4),
        "lon_std": round(float(lon.std()), 4),
    }


def compare_to_reference(
    filter_output: Dict[str, Any],
    reference_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare les statistiques des sites filtrés à celles d'un DataFrame de référence
    (ex: la totalité du jeu PA avant filtrage).

    Pour chaque colonne environnementale commune, calcule :
      - mean_filtered vs mean_reference
      - delta absolu et delta relatif (%)
      - si la différence est statistiquement notable (|delta_pct| > 10%)

    Parameters
    ----------
    filter_output : FilterOutput produit par Raihan
    reference_df  : DataFrame de référence (mêmes colonnes, toutes les observations)

    Returns
    -------
    {
      "ok": True,
      "n_filtered": int,
      "n_reference": int,
      "comparison": {
        col: {
          "mean_filtered": float, "mean_reference": float,
          "delta": float, "delta_pct": float, "notable": bool
        }, ...
      }
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if not isinstance(reference_df, pd.DataFrame):
        return _error(
            "Invalid reference_df",
            "reference_df doit être un pd.DataFrame.",
        )

    if filter_output["execution_status"]["is_empty"]:
        return {
            "ok":          True,
            "n_filtered":  0,
            "n_reference": len(reference_df),
            "comparison":  {},
            "warning":     "Aucun site filtré - comparaison impossible.",
        }

    df       = filter_output["df_combined"]
    col_fams = _detect_env_cols(df)
    env_cols = [c for c in _all_env_cols(col_fams) if c in df.columns and c in reference_df.columns]

    comparison: Dict[str, Any] = {}
    for col in env_cols:
        filt_vals = pd.to_numeric(df[col], errors="coerce").dropna()
        ref_vals  = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        if filt_vals.empty or ref_vals.empty:
            continue
        mean_f = float(filt_vals.mean())
        mean_r = float(ref_vals.mean())
        delta  = mean_f - mean_r
        delta_pct = round(100 * delta / mean_r, 2) if mean_r != 0 else float("inf")
        comparison[col] = {
            "mean_filtered":  round(mean_f, 4),
            "mean_reference": round(mean_r, 4),
            "delta":          round(delta, 4),
            "delta_pct":      delta_pct,
            "notable":        abs(delta_pct) > 10,
        }

    return {
        "ok":          True,
        "n_filtered":  int(len(df)),
        "n_reference": int(len(reference_df)),
        "filter_context": filter_output["filter_context"],
        "comparison":  comparison,
    }


def species_composition(
    filter_output: Dict[str, Any],
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Composition en espèces de la zone filtrée : quelles espèces dominent,
    lesquelles sont rares, lesquelles sont exclusives à cette zone.

    Parameters
    ----------
    top_k : nombre d'espèces les plus fréquentes à retourner

    Returns
    -------
    {
      "ok": True,
      "n_species_total": int,
      "n_sites": int,
      "top_species": [
        {"species_id": int, "n_sites": int, "prevalence_pct": float}, ...
      ],
      "rare_species": [{"species_id": int, "n_sites": int}, ...],
      "prevalence_summary": {mean, std, min, median, max},
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_species_total": 0, "n_sites": 0,
                "top_species": [], "rare_species": [], "prevalence_summary": {}}

    df = filter_output["df_combined"]

    if "speciesId" not in df.columns or "surveyId" not in df.columns:
        return _error(
            "Missing columns",
            "df_combined doit contenir speciesId et surveyId.",
        )

    n_sites = int(df["surveyId"].nunique())

    # Nombre de sites où chaque espèce est observée
    species_counts = (
        df.groupby("speciesId")["surveyId"]
        .nunique()
        .sort_values(ascending=False)
    )

    n_species = int(len(species_counts))
    prevalence = species_counts / n_sites * 100  # en %

    # Top K espèces
    top = [
        {
            "species_id":     int(sp),
            "n_sites":        int(cnt),
            "prevalence_pct": round(float(prevalence[sp]), 2),
        }
        for sp, cnt in species_counts.head(max(1, int(top_k))).items()
    ]

    # Espèces rares : présentes sur 1 seul site
    rare_mask  = species_counts == 1
    rare = [
        {"species_id": int(sp), "n_sites": 1}
        for sp in species_counts[rare_mask].index.tolist()
    ]

    # Résumé de la distribution de prévalence
    prev_arr = prevalence.values.astype(float)
    q25, q50, q75 = np.percentile(prev_arr, [25, 50, 75])
    prevalence_summary = {
        "mean":   round(float(prev_arr.mean()), 4),
        "std":    round(float(prev_arr.std()),  4),
        "min":    round(float(prev_arr.min()),  4),
        "q25":    round(float(q25), 4),
        "median": round(float(q50), 4),
        "q75":    round(float(q75), 4),
        "max":    round(float(prev_arr.max()),  4),
    }

    return {
        "ok":               True,
        "n_species_total":  n_species,
        "n_sites":          n_sites,
        "n_rare_species":   len(rare),
        "top_species":      top,
        "rare_species":     rare[:50],   # plafonné pour éviter les réponses énormes
        "prevalence_summary": prevalence_summary,
    }


def diversity_indices(filter_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Indices de diversité écologique par site et résumé global.

    Calcule pour chaque surveyId :
      - Shannon  H = -sum(p * log(p))   [0 = monoculture, max = ln(S)]
      - Simpson  D = 1 - sum(p²)        [0 = monoculture, 1 = max diversité]
      - Richesse S = nombre d'espèces distinctes

    Puis retourne les statistiques descriptives de ces indices sur tous les sites.

    Returns
    -------
    {
      "ok": True,
      "n_sites": int,
      "shannon": {mean, std, min, median, max},
      "simpson": {mean, std, min, median, max},
      "richness": {mean, std, min, median, max},
      "top_diverse_sites":  [{"surveyId", "shannon", "simpson", "richness"}, ...],
      "top_uniform_sites":  [{"surveyId", "shannon", "simpson", "richness"}, ...],
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_sites": 0, "shannon": {}, "simpson": {}, "richness": {}}

    df = filter_output["df_combined"]

    if "speciesId" not in df.columns or "surveyId" not in df.columns:
        return _error(
            "Missing columns",
            "df_combined doit contenir speciesId et surveyId.",
        )

    records = []
    for sid, grp in df.groupby("surveyId"):
        counts = grp["speciesId"].value_counts().values.astype(float)
        s      = len(counts)
        total  = counts.sum()
        if total == 0:
            continue
        p       = counts / total
        shannon = float(-np.sum(p * np.log(p + 1e-12)))
        simpson = float(1.0 - np.sum(p ** 2))
        records.append({
            "surveyId": int(sid),
            "shannon":  round(shannon, 4),
            "simpson":  round(simpson, 4),
            "richness": int(s),
        })

    if not records:
        return _error("No data", "Aucun site avec des espèces valides.")

    idx_df = pd.DataFrame(records)

    def _summarize(col: str) -> Dict[str, float]:
        arr = idx_df[col].values.astype(float)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return {
            "mean":   round(float(arr.mean()), 4),
            "std":    round(float(arr.std()),  4),
            "min":    round(float(arr.min()),  4),
            "q25":    round(float(q25), 4),
            "median": round(float(q50), 4),
            "q75":    round(float(q75), 4),
            "max":    round(float(arr.max()),  4),
        }

    top_div  = idx_df.nlargest(10,  "shannon").to_dict(orient="records")
    top_unif = idx_df.nsmallest(10, "shannon").to_dict(orient="records")

    return {
        "ok":               True,
        "n_sites":          int(len(idx_df)),
        "shannon":          _summarize("shannon"),
        "simpson":          _summarize("simpson"),
        "richness":         _summarize("richness"),
        "top_diverse_sites":  top_div,
        "top_uniform_sites":  top_unif,
    }


def species_env_correlation(
    filter_output: Dict[str, Any],
    top_k_species: int = 10,
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Corrélation entre la présence/absence des espèces les plus fréquentes
    et les variables environnementales.

    Pour chaque espèce parmi les top_k_species les plus prévalentes :
      - construit un vecteur binaire présence/absence par site
      - corrèle ce vecteur avec chaque variable env (moyenne par site)
      - retourne les variables les plus prédictives (|r| max)

    Parameters
    ----------
    top_k_species : nombre d'espèces à analyser
    method        : "pearson" | "spearman"

    Returns
    -------
    {
      "ok": True,
      "method": str,
      "n_sites": int,
      "species": [
        {
          "species_id": int,
          "prevalence_pct": float,
          "top_predictors": [
            {"env_var": str, "r": float, "direction": "positive"|"negative"}, ...
          ]
        }, ...
      ]
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "method": method, "n_sites": 0, "species": []}

    df = filter_output["df_combined"]

    if "speciesId" not in df.columns or "surveyId" not in df.columns:
        return _error(
            "Missing columns",
            "df_combined doit contenir speciesId et surveyId.",
        )

    # Variables environnementales disponibles
    col_fams = _detect_env_cols(df)
    env_cols = [c for c in _all_env_cols(col_fams) if c in df.columns]

    if not env_cols:
        return _error("No env columns", "Aucune colonne environnementale détectée.")

    # Table : une ligne par site, colonnes = moyenne des variables env
    site_env = (
        df.groupby("surveyId")[env_cols]
        .mean()
        .apply(pd.to_numeric, errors="coerce")
    )
    n_sites = len(site_env)

    if n_sites < 5:
        return _error("Not enough sites", "Minimum 5 sites requis pour la corrélation.")

    # Top K espèces par prévalence
    species_counts = df.groupby("speciesId")["surveyId"].nunique().sort_values(ascending=False)
    top_species    = species_counts.head(max(1, int(top_k_species))).index.tolist()

    results = []
    for sp in top_species:
        # Vecteur binaire présence/absence par site
        sites_with_sp = set(df[df["speciesId"] == sp]["surveyId"].unique())
        presence = pd.Series(
            [1 if sid in sites_with_sp else 0 for sid in site_env.index],
            index=site_env.index,
            dtype=float,
        )

        # Corrélation avec chaque variable env
        correlations = []
        for col in env_cols:
            env_vals = site_env[col].dropna()
            common   = presence.index.intersection(env_vals.index)
            if len(common) < 5:
                continue
            p_aligned   = presence.loc[common]
            env_aligned = env_vals.loc[common]
            if env_aligned.std() == 0:
                continue
            r = p_aligned.corr(env_aligned, method=method)
            if pd.isna(r):
                continue
            correlations.append({
                "env_var":   col,
                "r":         round(float(r), 4),
                "direction": "positive" if r > 0 else "negative",
            })

        correlations.sort(key=lambda x: abs(x["r"]), reverse=True)

        results.append({
            "species_id":      int(sp),
            "n_sites_present": int(species_counts[sp]),
            "prevalence_pct":  round(100 * species_counts[sp] / n_sites, 2),
            "top_predictors":  correlations[:5],
        })

    return {
        "ok":      True,
        "method":  method,
        "n_sites": n_sites,
        "species": results,
    }


def top_drivers(
    filter_output: Dict[str, Any],
    top_k_species: int = 10,
    top_k_vars: int = 8,
) -> Dict[str, Any]:
    """
    Résumé des 'Top Drivers' environnementaux pour la zone filtrée.

    Agrège trois signaux complémentaires pour identifier les variables
    qui structurent le mieux les communautés végétales de la zone :

      1. Variance relative : variables les plus discriminantes au sein de la zone
         (std_zone / std_global - élevé = variable qui varie beaucoup dans la zone)

      2. Corrélation avec la richesse spécifique : variables dont la valeur
         prédit le nombre d'espèces par site (r de Pearson)

      3. Corrélation multi-espèces : variables les plus souvent citées comme
         top predictors dans species_env_correlation (vote agrégé)

    Retourne un score composite normalisé [0-100] par variable
    et une interprétation lisible pour le LLM.

    Returns
    -------
    {
      "ok": True,
      "n_sites": int,
      "n_env_vars": int,
      "drivers": [
        {
          "rank": int,
          "env_var": str,
          "family": str,            <- BioClim / Sol / Elevation / LandCover / Autre
          "composite_score": float, <- 0-100, plus haut = plus influent
          "variance_score": float,  <- contribution de la variance relative
          "richness_corr": float,   <- r avec la richesse specifique (Pearson)
          "species_votes": int,     <- nb d'especes pour lesquelles c'est un top predictor
          "direction": str,         <- "positive" / "negative" / "mixed" vs richesse
          "interpretation": str,    <- phrase lisible par le LLM
        }, ...
      ],
      "family_importance": {        <- % de score composite par famille
        "BioClim": float,
        "Sol": float,
        "Elevation": float,
        ...
      },
      "llm_summary": str,           <- paragraphe de synthese pret pour le LLM
    }
    """
    err_msg = _validate_filter_output(filter_output)
    if err_msg:
        return _error("Invalid FilterOutput", err_msg)

    if filter_output["execution_status"]["is_empty"]:
        return {"ok": True, "n_sites": 0, "n_env_vars": 0, "drivers": [],
                "family_importance": {}, "llm_summary": "Aucun site filtre."}

    df = filter_output["df_combined"]

    if "surveyId" not in df.columns or "speciesId" not in df.columns:
        return _error("Missing columns",
                      "df_combined doit contenir surveyId et speciesId.")

    # ── Variables environnementales disponibles ──────────────────────────────
    col_fams = _detect_env_cols(df)
    env_cols = [c for c in _all_env_cols(col_fams) if c in df.columns]

    if len(env_cols) < 2:
        return _error("Not enough env columns",
                      "Au moins 2 colonnes environnementales requises.")

    # ── Table site-level : une ligne par surveyId ────────────────────────────
    site_df = df.drop_duplicates(subset=["surveyId"]).set_index("surveyId")
    num_env = site_df[env_cols].apply(pd.to_numeric, errors="coerce")

    # Richesse par site
    richness = df.groupby("surveyId")["speciesId"].nunique().rename("richness")
    num_env  = num_env.join(richness, how="left")

    n_sites = int(len(num_env))

    # ── Signal 1 : variance relative (std zone / moyenne globale) ────────────
    std_zone  = num_env[env_cols].std()
    mean_zone = num_env[env_cols].mean().replace(0, float("nan"))
    var_score = (std_zone / mean_zone.abs()).fillna(0).abs()
    # Normaliser 0-1
    var_max = var_score.max()
    var_score_norm = (var_score / var_max) if var_max > 0 else var_score

    # ── Signal 2 : corrélation avec la richesse ───────────────────────────────
    richness_corr = {}
    for col in env_cols:
        env_vals = num_env[col].dropna()
        rich_vals = num_env["richness"].loc[env_vals.index].dropna()
        common = env_vals.index.intersection(rich_vals.index)
        if len(common) < 5:
            richness_corr[col] = 0.0
            continue
        r = env_vals.loc[common].corr(rich_vals.loc[common])
        richness_corr[col] = float(r) if not pd.isna(r) else 0.0

    richness_corr_series = pd.Series(richness_corr)
    rc_abs = richness_corr_series.abs()
    rc_max = rc_abs.max()
    rc_norm = (rc_abs / rc_max) if rc_max > 0 else rc_abs

    # ── Signal 3 : votes multi-espèces ───────────────────────────────────────
    sp_env_result = species_env_correlation(
        filter_output, top_k_species=top_k_species, method="pearson"
    )
    species_votes: Dict[str, int] = {col: 0 for col in env_cols}
    if sp_env_result.get("ok"):
        for sp_info in sp_env_result.get("species", []):
            for pred in sp_info.get("top_predictors", [])[:3]:
                col = pred["env_var"]
                if col in species_votes:
                    species_votes[col] += 1

    votes_series = pd.Series(species_votes)
    votes_max = votes_series.max()
    votes_norm = (votes_series / votes_max) if votes_max > 0 else votes_series

    # ── Score composite : pondération des 3 signaux ───────────────────────────
    # Variance 30% + corrélation richesse 40% + votes espèces 30%
    composite = (
        0.30 * var_score_norm.reindex(env_cols, fill_value=0) +
        0.40 * rc_norm.reindex(env_cols, fill_value=0) +
        0.30 * votes_norm.reindex(env_cols, fill_value=0)
    )
    composite_norm = (composite / composite.max() * 100).round(2) if composite.max() > 0 else composite

    # ── Classification par famille ────────────────────────────────────────────
    def _family(col: str) -> str:
        n = col.lower()
        if n.startswith("bio"):
            return "BioClim"
        if "soil" in n or n.startswith("soilgrid") or n.startswith("phh2o") \
                or n.startswith("bdod") or n.startswith("soc") or n.startswith("cec"):
            return "Sol"
        if "elev" in n or "altitude" in n:
            return "Elevation"
        if "landcover" in n or "land_cover" in n:
            return "LandCover"
        if "human" in n or "footprint" in n:
            return "Human Footprint"
        return "Autre"

    def _interpretation(col: str, r: float, score: float, votes: int) -> str:
        fam   = _family(col)
        direc = "positivement" if r > 0 else "negativement"
        strength = "fortement" if abs(r) > 0.3 else "moderement" if abs(r) > 0.1 else "faiblement"
        vote_str = f", predicateur pour {votes} espece(s) dominante(s)" if votes > 0 else ""
        return (
            f"{col} ({fam}) est {strength} correlee {direc} "
            f"a la richesse specifique (r={r:.2f}){vote_str}."
        )

    # ── Construction de la liste triée ───────────────────────────────────────
    ranked = composite_norm.sort_values(ascending=False)
    drivers = []
    for rank, (col, score) in enumerate(ranked.head(top_k_vars).items(), start=1):
        r     = richness_corr.get(col, 0.0)
        votes = int(species_votes.get(col, 0))
        direc = "positive" if r > 0.05 else "negative" if r < -0.05 else "mixed"
        drivers.append({
            "rank":            rank,
            "env_var":         col,
            "family":          _family(col),
            "composite_score": float(score),
            "variance_score":  round(float(var_score_norm.get(col, 0) * 100), 2),
            "richness_corr":   round(r, 4),
            "species_votes":   votes,
            "direction":       direc,
            "interpretation":  _interpretation(col, r, float(score), votes),
        })

    # ── Importance par famille ────────────────────────────────────────────────
    family_scores: Dict[str, float] = {}
    for col, score in composite_norm.items():
        fam = _family(col)
        family_scores[fam] = family_scores.get(fam, 0.0) + float(score)
    total_fam = sum(family_scores.values()) + 1e-8
    family_importance = {
        k: round(v / total_fam * 100, 1)
        for k, v in sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
    }

    # ── Résumé textuel pour le LLM ────────────────────────────────────────────
    top3 = drivers[:3]
    family_top = max(family_importance, key=family_importance.get) if family_importance else "inconnue"
    llm_summary = (
        f"Sur {n_sites} sites filtres, les variables environnementales "
        f"les plus structurantes sont : "
        + ", ".join(f"{d['env_var']} (score={d['composite_score']:.0f}/100, "
                    f"r richesse={d['richness_corr']:+.2f})" for d in top3)
        + f". La famille la plus influente est '{family_top}' "
        f"({family_importance.get(family_top, 0):.0f}% du score composite). "
        + ("Ces drivers sont bases sur la variance intra-zone, la correlation "
           "avec la richesse specifique, et le vote multi-especes.")
    )

    return {
        "ok":               True,
        "n_sites":          n_sites,
        "n_env_vars":       len(env_cols),
        "drivers":          drivers,
        "family_importance": family_importance,
        "llm_summary":      llm_summary,
    }
