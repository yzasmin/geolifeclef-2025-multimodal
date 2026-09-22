"""
tools_data.py - Raihan
=======================
Deux interfaces en un fichier :

  A) Interface MCP (Guilhem) - fonctions JSON-sérialisables pour le serveur MCP
     filter_by_region, filter_by_region_name, get_random_images,
     select_model, get_server_status
     → retournent {"ok": True, ...}

  B) Interface Malala - FilterOutput pour le module statistiques
     apply_filter, build_filter_output
     → retournent {"df_combined": pd.DataFrame, "filter_context": dict,
                   "file_registry": dict, "execution_status": dict}

Variables d'environnement :
  DATA_CSV_PATH   chemin du CSV principal (pour les tests unitaires)
  TIFF_ROOT       racine des TIFF (pour les tests unitaires)
  DATA_DIR        racine des données réelles sur RTX5
                  (défaut : ./data)

Usage RTX5 :
  DATA_DIR=./data python tools_data.py
"""

from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────
#  CHEMINS - DATA_DIR pour les vraies données RTX5
# ──────────────────────────────────────────────────────────────

DATA_DIR    = Path(os.environ.get("DATA_DIR",
              "data"))
ENV_DIR     = DATA_DIR / "EnvironmentalValues"
PATCHES_DIR = DATA_DIR / "SatelitePatches" / "PA-train"

# Chemins CSV environnementaux
_BIOCLIM_CSV   = ENV_DIR / "ClimateAverage_1981-2010" / "GLC25-PA-train-bioclimatic.csv"
_SOILGRIDS_CSV = ENV_DIR / "SoilGrids"                / "GLC25-PA-train-soilgrids.csv"
_ELEVATION_CSV = ENV_DIR / "Elevation"                / "GLC25-PA-train-elevation.csv"
_LANDCOVER_CSV = ENV_DIR / "LandCover"                / "GLC25-PA-train-landcover.csv"
_HFP_CSV       = ENV_DIR / "HumanFootprint"           / "GLC25-PA-train-human_footprint.csv"

# ──────────────────────────────────────────────────────────────
#  RENOMMAGE DES COLONNES CSV → noms du contrat Malala
# ──────────────────────────────────────────────────────────────

# Bioclim : Bio1 → bio_1, ..., Bio19 → bio_19
_BIOCLIM_RENAME: dict[str, str] = {f"Bio{i}": f"bio_{i}" for i in range(1, 20)}

# SoilGrids : noms réels du CSV → noms du contrat
# (les noms exacts dépendent de la version ; on couvre les deux conventions)
_SOIL_RENAME: dict[str, str] = {
    # Convention GLC25 / SoilGrids ISRIC
    "bdod": "soil_bdod", "cec": "soil_cec",   "cfvo": "soil_cfvo",
    "clay": "soil_clay", "nitrogen": "soil_nitrogen",
    "phh2o": "soil_pH",  "sand": "soil_sand", "silt": "soil_silt",
    "soc":   "soil_soc",
    # Noms réels observés dans le CSV GeoLifeCLEF (préfixe Soilgrid-)
    "Soilgrid-bdod": "soil_bdod", "Soilgrid-cec": "soil_cec",
    "Soilgrid-cfvo": "soil_cfvo", "Soilgrid-clay": "soil_clay",
    "Soilgrid-nitrogen": "soil_nitrogen", "Soilgrid-phh2o": "soil_pH",
    "Soilgrid-sand": "soil_sand", "Soilgrid-silt": "soil_silt",
    "Soilgrid-soc": "soil_soc",
    # Variante éventuelle en minuscule
    "soilgrid-bdod": "soil_bdod", "soilgrid-cec": "soil_cec",
    "soilgrid-cfvo": "soil_cfvo", "soilgrid-clay": "soil_clay",
    "soilgrid-nitrogen": "soil_nitrogen", "soilgrid-phh2o": "soil_pH",
    "soilgrid-sand": "soil_sand", "soilgrid-silt": "soil_silt",
    "soilgrid-soc": "soil_soc",
    # Variantes possibles
    "pH": "soil_pH",     "Clay": "soil_clay", "Sand": "soil_sand",
    "Silt": "soil_silt", "SOC": "soil_soc",   "Nitrogen": "soil_nitrogen",
    "CEC": "soil_cec",   "BDOD": "soil_bdod", "CFVO": "soil_cfvo",
}

# Élévation : Elevation → elev
_ELEV_RENAME: dict[str, str] = {"Elevation": "elev", "elevation": "elev", "ELEV": "elev"}

# ══════════════════════════════════════════════════════════════
#  SECTION A - INTERFACE MCP (Guilhem)
#  Toutes les fonctions ci-dessous respectent le contrat de
#  test_tools_modules.py (retour {"ok": True, ...})
# ══════════════════════════════════════════════════════════════

# ── Variables d'env pour les tests unitaires ─────────────────
DATA_CSV_ENV    = "DATA_CSV_PATH"
TIFF_ROOT_ENV   = "TIFF_ROOT"
DEFAULT_MODEL_ENV = "ACTIVE_MODEL"
MODEL_STATE_PATH_ENV = "MCP_MODEL_STATE_PATH"


def _default_model_state_path() -> Path:
    # Par défaut: fichier local du repo (stable entre sessions/process MCP).
    return Path(__file__).resolve().parent.parent / ".mcp_model_state.json"


def _model_state_path() -> Path:
    raw = (os.getenv(MODEL_STATE_PATH_ENV, "") or "").strip()
    return Path(raw).expanduser() if raw else _default_model_state_path()


def _load_selected_model_from_disk() -> Optional[str]:
    state_path = _model_state_path()
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("selected_model")
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _save_selected_model_to_disk(model_name: str) -> tuple[bool, Optional[str]]:
    state_path = _model_state_path()
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"selected_model": model_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True, None
    except Exception as exc:
        return False, str(exc)

_initial_selected_model = (
    (os.getenv(DEFAULT_MODEL_ENV, "") or "").strip()
    or _load_selected_model_from_disk()
    or "v1"
)
_STATE: dict[str, Any] = {"selected_model": _initial_selected_model}
_LOCK  = threading.Lock()
_CACHE: dict[str, Any] = {"csv_path": None, "mtime_ns": None, "df": None}

_COLUMN_ALIASES: dict[str, list[str]] = {
    "survey_id":  ["surveyid", "survey_id", "id"],
    "species_id": ["spid", "speciesid", "species_id", "taxonid"],
    "lat":        ["lat", "latitude", "decimallatitude"],
    "lon":        ["lon", "lng", "longitude", "decimallongitude"],
    "region":     ["region", "admin1", "state", "province", "area"],
    "country":    ["country", "countryname", "nation"],
}


def _error(error: str, hint: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": error, "hint": hint}
    payload.update(extra)
    return payload


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _record_to_json(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _safe_value(value) for key, value in record.items()}


def _records_to_json(df: pd.DataFrame, limit: int = 1000) -> list[dict[str, Any]]:
    limited = df.head(max(0, int(limit)))
    return [_record_to_json(row) for row in limited.to_dict(orient="records")]


def _normalize_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    normalized_to_original = {_normalize_key(col): col for col in df.columns}
    for alias in aliases:
        candidate = normalized_to_original.get(_normalize_key(alias))
        if candidate is not None:
            return candidate
    return None


def _resolve_standard_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    return {name: _resolve_column(df, aliases)
            for name, aliases in _COLUMN_ALIASES.items()}


def _load_dataframe() -> tuple[Optional[pd.DataFrame], Optional[dict[str, Any]]]:
    """
    Charge le CSV défini par DATA_CSV_PATH (tests) ou DATA_DIR (production).
    Utilise un cache thread-safe invalidé sur modification du fichier.
    """
    # 1. Priorité à DATA_CSV_PATH (tests unitaires de Guilhem)
    csv_path_str = os.getenv(DATA_CSV_ENV)

    # 2. Fallback vers le CSV principal sur RTX5
    if not csv_path_str:
        meta_path = DATA_DIR / "GLC25_PA_metadata_train.csv"
        if meta_path.exists():
            csv_path_str = str(meta_path)

    if not csv_path_str:
        return None, _error(
            "No CSV found",
            "Set DATA_CSV_PATH or DATA_DIR to point to your dataset.",
        )

    csv_file = Path(csv_path_str)
    if not csv_file.exists():
        return None, _error(
            "CSV file not found",
            "Verify DATA_CSV_PATH or DATA_DIR.",
            csv_path=str(csv_file),
        )

    mtime_ns = csv_file.stat().st_mtime_ns
    with _LOCK:
        if (
            _CACHE["df"] is not None
            and _CACHE["csv_path"] == str(csv_file)
            and _CACHE["mtime_ns"] == mtime_ns
        ):
            return _CACHE["df"], None

        try:
            df = pd.read_csv(csv_file, low_memory=False)
        except Exception as exc:
            return None, _error(
                "Failed to read CSV", "Check CSV format.",
                details=str(exc), csv_path=str(csv_file),
            )

        _CACHE["csv_path"] = str(csv_file)
        _CACHE["mtime_ns"] = mtime_ns
        _CACHE["df"]       = df
        return df, None


def _normalize_survey_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        maybe_int = text[:-2]
        if maybe_int.isdigit():
            text = maybe_int
    return text


def _get_tiff_root() -> Optional[Path]:
    """Retourne la racine TIFF : TIFF_ROOT (tests) ou PATCHES_DIR (RTX5)."""
    root = os.getenv(TIFF_ROOT_ENV)
    if root:
        return Path(root)
    if PATCHES_DIR.exists():
        return PATCHES_DIR
    return None


def _survey_to_relative_tiff_path(survey_id: Any, extension: str = ".tiff") -> Path:
    sid = _normalize_survey_id(survey_id) or ""
    if len(sid) >= 4:
        return Path(sid[-2:]) / sid[-4:-2] / f"{sid}{extension}"
    return Path(sid) / f"{sid}{extension}"


def _find_tiff_path(survey_id: Any) -> Optional[Path]:
    tiff_root = _get_tiff_root()
    if tiff_root is None:
        return None
    sid = _normalize_survey_id(survey_id)
    if sid is None:
        return None
    for ext in (".tiff", ".tif"):
        candidate = tiff_root / _survey_to_relative_tiff_path(sid, extension=ext)
        if candidate.exists():
            return candidate
    return None


# ── Fonctions MCP publiques ───────────────────────────────────

def filter_by_region(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit: int = 1000,
) -> dict[str, Any]:
    """
    Filtre les observations par boîte GPS [min_lat, max_lat] × [min_lon, max_lon].

    Returns {"ok": True, "matched_count": int, "returned_count": int, "records": list}
    """
    if min_lat > max_lat or min_lon > max_lon:
        return _error("Invalid bounding box",
                      "Expected min <= max for both lat and lon.")

    df, err = _load_dataframe()
    if err:
        return err
    assert df is not None

    columns = _resolve_standard_columns(df)
    lat_col, lon_col = columns["lat"], columns["lon"]
    if lat_col is None or lon_col is None:
        return _error("Missing GPS columns",
                      "CSV must contain lat and lon columns.",
                      detected_columns=columns)

    lat  = pd.to_numeric(df[lat_col], errors="coerce")
    lon  = pd.to_numeric(df[lon_col], errors="coerce")
    mask = lat.between(float(min_lat), float(max_lat)) & \
           lon.between(float(min_lon), float(max_lon))
    filtered = df[mask]

    return {
        "ok": True,
        "query": {
            "min_lat": float(min_lat), "max_lat": float(max_lat),
            "min_lon": float(min_lon), "max_lon": float(max_lon),
            "limit":   int(limit),
        },
        "matched_count":  int(len(filtered)),
        "returned_count": int(min(len(filtered), max(0, int(limit)))),
        "records":        _records_to_json(filtered, limit=limit),
    }


def filter_by_region_name(
    region_name: str,
    limit: int = 1000,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """
    Filtre les observations par nom de région/pays (substring, insensible à la casse).

    Returns {"ok": True, "matched_count": int, "returned_count": int, "records": list}
    """
    if not region_name or not region_name.strip():
        return _error("Empty region_name", "Provide a non-empty region name.")

    df, err = _load_dataframe()
    if err:
        return err
    assert df is not None

    columns    = _resolve_standard_columns(df)
    region_col = columns["region"] or columns["country"]
    if region_col is None:
        return _error("Missing region column",
                      "CSV should expose region, country, or equivalent.",
                      detected_columns=columns)

    series  = df[region_col].astype(str)
    pattern = region_name.strip()
    mask    = series.str.contains(pattern, regex=False, na=False) \
              if case_sensitive \
              else series.str.lower().str.contains(pattern.lower(),
                                                   regex=False, na=False)
    filtered = df[mask]

    return {
        "ok": True,
        "query": {
            "region_name":    region_name,
            "case_sensitive": bool(case_sensitive),
            "limit":          int(limit),
            "matched_column": region_col,
        },
        "matched_count":  int(len(filtered)),
        "returned_count": int(min(len(filtered), max(0, int(limit)))),
        "records":        _records_to_json(filtered, limit=limit),
    }


def get_random_images(
    n: int = 10,
    seed: Optional[int] = None,
    region_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Retourne N images Sentinel-2 aléatoires (chemins TIFF).
    seed permet la reproductibilité.

    Returns {"ok": True, "returned": int, "images": list[{"survey_id", "image_path"}]}
    """
    if int(n) <= 0:
        return _error("Invalid n", "n must be a strictly positive integer.")

    tiff_root = _get_tiff_root()
    if tiff_root is None:
        return _error("TIFF_ROOT unavailable",
                      "Set TIFF_ROOT or DATA_DIR pointing to the TIFF directory.")

    df, err = _load_dataframe()
    if err:
        return err
    assert df is not None

    columns    = _resolve_standard_columns(df)
    survey_col = columns["survey_id"]
    if survey_col is None:
        return _error("Missing survey id column",
                      "CSV must include surveyId.",
                      detected_columns=columns)

    work_df = df
    if region_name:
        region_col = columns["region"] or columns["country"]
        if region_col:
            mask    = work_df[region_col].astype(str).str.lower()\
                        .str.contains(region_name.lower(), regex=False, na=False)
            work_df = work_df[mask]

    survey_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in work_df[survey_col].tolist():
        sid = _normalize_survey_id(raw_id)
        if sid is None or sid in seen:
            continue
        seen.add(sid)
        survey_ids.append(sid)

    rng = random.Random(seed)
    rng.shuffle(survey_ids)

    images: list[dict[str, Any]] = []
    for sid in survey_ids:
        candidate = _find_tiff_path(sid)
        if candidate is None:
            continue
        images.append({
            "survey_id":     sid,
            "image_path":    str(candidate),
            "relative_path": str(candidate.relative_to(tiff_root)),
        })
        if len(images) >= int(n):
            break

    return {
        "ok":          True,
        "requested":   int(n),
        "returned":    int(len(images)),
        "seed":        seed,
        "region_name": region_name,
        "images":      images,
    }


def select_model(model_name: str) -> dict[str, Any]:
    """
    Sélectionne le modèle actif pour les prédictions et explications.
    Modèles : 'v1' (0.230), 'v4b' (0.229), 'v6' (0.230+).
    """
    if not model_name or not model_name.strip():
        return _error("Empty model_name",
                      "Provide a model tag such as v1, v4b, or v6.")
    previous             = _STATE.get("selected_model")
    selected = model_name.strip()
    _STATE["selected_model"] = selected
    persisted, persistence_error = _save_selected_model_to_disk(selected)
    payload = {
        "ok":             True,
        "previous_model": previous,
        "selected_model": selected,
        "state_file":     str(_model_state_path()),
        "state_persisted": persisted,
    }
    if persistence_error:
        payload["state_persistence_error"] = persistence_error
    return payload


def get_server_status() -> dict[str, Any]:
    """
    État du serveur MCP : données chargées, modèle sélectionné, GPU disponible.
    """
    csv_path  = os.getenv(DATA_CSV_ENV) or str(DATA_DIR / "GLC25_PA_metadata_train.csv")
    tiff_root = _get_tiff_root()

    df, err      = _load_dataframe()
    csv_available = err is None and df is not None
    row_count     = int(len(df)) if df is not None else 0
    columns       = _resolve_standard_columns(df) if df is not None else {}
    persisted_model = _load_selected_model_from_disk()
    if persisted_model:
        _STATE["selected_model"] = persisted_model

    return {
        "ok":                True,
        "csv_available":     bool(csv_available),
        "csv_path":          csv_path,
        "csv_error":         None if csv_available else err,
        "tiff_root_available": bool(tiff_root is not None and tiff_root.exists()),
        "tiff_root":         str(tiff_root) if tiff_root is not None else None,
        "row_count":         row_count,
        "selected_model":    _STATE.get("selected_model"),
        "state_file":        str(_model_state_path()),
        "detected_columns":  columns,
    }


def _reset_cache_for_tests() -> None:
    """Réinitialise le cache - appelé par setUp() dans les tests."""
    with _LOCK:
        _CACHE["csv_path"] = None
        _CACHE["mtime_ns"] = None
        _CACHE["df"]       = None


# ══════════════════════════════════════════════════════════════
#  SECTION B - INTERFACE MALALA (FilterOutput)
#  Fonctions qui chargent, mergent et filtrent les vraies données
#  pour passer à tools_stats.py
# ══════════════════════════════════════════════════════════════

# Type alias documentaire
FilterOutput = Dict[str, Any]


def _get_tiff_abs_path(survey_id: int) -> Optional[str]:
    """
    Retourne le chemin absolu du TIFF pour un surveyId, ou None si absent.
    Utilise PATCHES_DIR ou TIFF_ROOT selon le contexte.
    """
    sid  = str(survey_id)
    root = _get_tiff_root()
    if root is None:
        return None
    d1 = sid[-2:]   if len(sid) >= 2 else sid.zfill(2)
    d2 = sid[-4:-2] if len(sid) >= 4 else "00"
    for ext in (".tiff", ".tif"):
        path = root / d1 / d2 / f"{sid}{ext}"
        if path.exists():
            return str(path.resolve())
    return None


def _load_env_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge et renomme les 3 CSV environnementaux principaux.
    Retourne (bioclim, soil, elev) avec les noms du contrat Malala.
    """
    # Bioclim
    bioclim = pd.read_csv(_BIOCLIM_CSV)
    bioclim.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    bioclim.rename(columns=_BIOCLIM_RENAME, inplace=True)

    # SoilGrids
    soil = pd.read_csv(_SOILGRIDS_CSV)
    soil.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    soil.rename(columns=_SOIL_RENAME, inplace=True)

    # Élévation
    elev_df = pd.read_csv(_ELEVATION_CSV)
    elev_df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    elev_df.rename(columns=_ELEV_RENAME, inplace=True)

    return bioclim, soil, elev_df


def _build_merged_df() -> pd.DataFrame:
    """
    Charge et merge les métadonnées PA + les 3 CSV environnementaux.
    Une ligne par (surveyId, speciesId) - même structure que GLC25_PA_metadata_train.csv.
    """
    meta    = pd.read_csv(DATA_DIR / "GLC25_PA_metadata_train.csv")
    bioclim, soil, elev_df = _load_env_data()

    # speciesId : float → int (certaines lignes peuvent avoir NaN → on les supprime)
    meta = meta.dropna(subset=["speciesId"])
    meta["speciesId"] = meta["speciesId"].astype(int)
    meta["surveyId"]  = meta["surveyId"].astype(int)

    # Merge environnemental
    env = (
        bioclim
        .merge(soil,    on="surveyId", how="outer")
        .merge(elev_df, on="surveyId", how="outer")
    )
    env["surveyId"] = env["surveyId"].astype(int)

    # Merge final
    df = meta.merge(env, on="surveyId", how="inner")

    # Supprimer les doublons stricts
    df = df.drop_duplicates(subset=["surveyId", "speciesId"])
    df = df.reset_index(drop=True)

    return df


def build_filter_output(
    df_combined:    pd.DataFrame,
    filter_context: dict[str, Any],
    file_registry:  dict[int, Optional[str]],
) -> FilterOutput:
    """
    Construit et valide un FilterOutput conforme au contrat Malala.

    Vérifie :
      - df_combined est un DataFrame avec les colonnes obligatoires
      - filter_context est sérialisable (pas de numpy/DataFrame)
      - file_registry couvre tous les surveyId de df_combined
      - execution_status est cohérent

    Raises ValueError si le contrat est violé.
    """
    # Vérification colonnes obligatoires
    required = {"surveyId", "speciesId", "lat", "lon"}
    missing  = required - set(df_combined.columns)
    if missing:
        raise ValueError(f"df_combined manque les colonnes : {missing}")

    # Vérification au moins une colonne environnementale
    env_cols = [c for c in df_combined.columns
                if c.startswith("bio_") or c.startswith("soil_") or c == "elev"]
    if not env_cols:
        raise ValueError("df_combined ne contient aucune colonne environnementale "
                         "(bio_*, soil_*, elev).")

    # Index RangeIndex
    if not isinstance(df_combined.index, pd.RangeIndex):
        df_combined = df_combined.reset_index(drop=True)

    # Cohérence file_registry
    survey_ids_in_df = set(df_combined["surveyId"].unique().tolist())
    for sid in survey_ids_in_df:
        if sid not in file_registry:
            raise ValueError(
                f"surveyId {sid} présent dans df_combined "
                f"mais absent de file_registry."
            )

    count    = len(df_combined)
    is_empty = count == 0

    return {
        "df_combined":      df_combined,
        "filter_context":   filter_context,
        "file_registry":    file_registry,
        "execution_status": {"count": count, "is_empty": is_empty},
    }


def apply_filter(
    region:        Optional[str]   = None,
    elevation_min: float           = 0.0,
    elevation_max: float           = 9000.0,
    bioclim_var:   Optional[str]   = None,
    bioclim_min:   Optional[float] = None,
    bioclim_max:   Optional[float] = None,
    country:       Optional[str]   = None,
    n_species_min: Optional[int]   = None,
) -> FilterOutput:
    """
    Fonction principale du contrat Malala.
    Charge, merge et filtre les données, puis retourne un FilterOutput validé.

    Parameters
    ----------
    region        : région biogéographique exacte du CSV
                    (ex: 'MEDITERRANEAN', 'ALPINE', 'ATLANTIC', 'CONTINENTAL'...)
    elevation_min : altitude minimale en mètres
    elevation_max : altitude maximale en mètres
    bioclim_var   : colonne BioClim à filtrer (ex: 'bio_12')
    bioclim_min   : seuil minimum pour bioclim_var
    bioclim_max   : seuil maximum pour bioclim_var
    country       : pays exact (ex: 'France', 'Italy', 'Denmark')
    n_species_min : nombre minimum d'espèces par surveyId

    Returns
    -------
    FilterOutput avec clés :
        df_combined      - pd.DataFrame enrichi (surveyId, speciesId, lat, lon,
                           bio_1…bio_19, soil_*, elev, region, country, ...)
        filter_context   - dict décrivant les filtres appliqués
        file_registry    - dict[int, str|None] surveyId → chemin TIFF absolu
        execution_status - {"count": int, "is_empty": bool}
    """
    # ── 1. Chargement et merge ──────────────────────────────────────────────
    df = _build_merged_df()

    # ── 2. Application des filtres + construction du contexte ───────────────
    context: dict[str, Any] = {}

    if region is not None:
        df = df[df["region"] == region]
        context["region"] = region

    if country is not None:
        df = df[df["country"].str.lower() == country.lower()]
        context["country"] = country

    if "elev" in df.columns:
        df = df[df["elev"].between(elevation_min, elevation_max)]
        context["elevation_min"] = elevation_min
        context["elevation_max"] = elevation_max

    if bioclim_var is not None and bioclim_var in df.columns:
        if bioclim_min is not None:
            df = df[df[bioclim_var] >= bioclim_min]
            context[f"{bioclim_var}_min"] = bioclim_min
        if bioclim_max is not None:
            df = df[df[bioclim_var] <= bioclim_max]
            context[f"{bioclim_var}_max"] = bioclim_max

    if n_species_min is not None:
        species_per_survey = df.groupby("surveyId")["speciesId"].nunique()
        valid_ids = species_per_survey[species_per_survey >= n_species_min].index
        df        = df[df["surveyId"].isin(valid_ids)]
        context["n_species_min"] = n_species_min

    if not context:
        context = {"region": "all", "filter_applied": False}

    df = df.reset_index(drop=True)

    # ── 3. Registre de fichiers ─────────────────────────────────────────────
    unique_ids = df["surveyId"].unique().tolist()
    registry: dict[int, Optional[str]] = {
        int(sid): _get_tiff_abs_path(int(sid)) for sid in unique_ids
    }

    # ── 4. Retour validé ────────────────────────────────────────────────────
    return build_filter_output(
        df_combined    = df,
        filter_context = context,
        file_registry  = registry,
    )


# ──────────────────────────────────────────────────────────────
#  TEST - lance :
#  DATA_DIR=./data python tools_data.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEST tools_data.py")
    print("=" * 60)

    # ── A) Interface MCP ───────────────────────────────────────
    print("\n── Interface MCP ──")
    print("[1] get_server_status()")
    status = get_server_status()
    for k, v in status.items():
        if k not in ("csv_error",):
            print(f"  {k:30s}: {v}")

    print("\n[2] filter_by_region_name('MEDITERRANEAN', limit=3)")
    r = filter_by_region_name("MEDITERRANEAN", limit=3)
    print(f"  matched_count={r.get('matched_count')}  ok={r.get('ok')}")

    print("\n[3] filter_by_region(43, 46, 2, 8, limit=3)")
    r = filter_by_region(43, 46, 2, 8, limit=3)
    print(f"  matched_count={r.get('matched_count')}  ok={r.get('ok')}")

    print("\n[4] get_random_images(n=2, seed=42)")
    imgs = get_random_images(n=2, seed=42)
    print(f"  returned={imgs.get('returned')}  ok={imgs.get('ok')}")

    print("\n[5] select_model('v6')")
    print(" ", select_model("v6"))

    # ── B) Interface Malala ────────────────────────────────────
    print("\n── Interface Malala (FilterOutput) ──")
    print("[6] apply_filter(region='MEDITERRANEAN', elevation_min=200)")
    try:
        out = apply_filter(region="MEDITERRANEAN", elevation_min=200.0)
        st  = out["execution_status"]
        df  = out["df_combined"]
        reg = out["file_registry"]
        print(f"  count={st['count']}  is_empty={st['is_empty']}")
        print(f"  colonnes df: {df.columns.tolist()[:10]}...")
        n_with_tiff = sum(1 for v in reg.values() if v is not None)
        print(f"  fichiers TIFF disponibles: {n_with_tiff}/{len(reg)}")
        print(f"  filter_context: {out['filter_context']}")
    except Exception as e:
        print(f"  ERREUR: {e}")

    print("\n[7] apply_filter() sans filtre - toutes les données")
    try:
        out = apply_filter()
        st  = out["execution_status"]
        print(f"  count={st['count']}  is_empty={st['is_empty']}")
    except Exception as e:
        print(f"  ERREUR: {e}")

    print("\n=== OK ===")
