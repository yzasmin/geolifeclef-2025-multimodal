"""
tools_explain.py - Explicabilité du modèle GeoLifeCLEF 2026
============================================================
Exposé comme outil MCP via server.py (Guilhem).
S'intègre sans conflit avec tools_data.py (Raihan) et tools_stats.py (Malala).

Fonctions MCP publiques :
  explain_model_prediction(survey_id, top_k, checkpoint_name)
  get_branch_importance(survey_id, checkpoint_name)
  get_attention_weights(survey_id, checkpoint_name)
  list_available_checkpoints()

Technique d'explicabilité :
  - Gradient × Input sur les features tabulaires (env + aux)
    → quelle variable Bio/Sol/Élévation a le plus influencé la prédiction
  - Hooks sur les TransformerEncoderLayer
    → quelle saison Landsat / quel mois bioclim a le plus pesé
  - Norme L2 des embeddings de chaque branche
    → quelle branche du modèle a le plus contribué à la décision finale

Variables d'environnement :
  DATA_DIR            racine des données (même que tools_data.py)
  CHECKPOINTS_DIR     dossier contenant fold0_best.pt … fold4_best.pt
                      (défaut : DATA_DIR/../artifacts_v4)
  EXPLAIN_DEVICE      "cpu" ou "cuda" (défaut : cpu pour la démo)
"""

from __future__ import annotations

import os
import sys
import re
import math
import warnings
import importlib.util
import io
import contextlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    "data"
))

# Les checkpoints peuvent être dans artifacts_v4 ou Challenge_Deep_Plant_Predict/artifacts
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_DIR_RAW = (os.environ.get("CHECKPOINTS_DIR", "") or "").strip()


def _checkpoint_search_paths() -> List[Path]:
    paths: List[Path] = []
    if _CHECKPOINTS_DIR_RAW:
        paths.append(Path(_CHECKPOINTS_DIR_RAW))
    paths.extend(
        [
            # Cas local étudiant: fold4_best.pt posé à la racine du repo.
            DATA_DIR.parent,
            _REPO_ROOT,
            # Cas serveur/entraînement.
            DATA_DIR.parent / "artifacts_v4",
            DATA_DIR.parent / "artifacts",
            DATA_DIR.parent / "checkpoints_5fold",
            (Path(os.environ.get("GLC_OUTPUT_DIR", "artifacts")) / "checkpoints_5fold"),
            Path("artifacts"),
        ]
    )
    return paths


_CKPT_CANDIDATES = _checkpoint_search_paths()
DEFAULT_EXPLAIN_CHECKPOINT_ENV = "DEFAULT_EXPLAIN_CHECKPOINT"
EXPLAIN_SAVE_PLOTS_ENV = "EXPLAIN_SAVE_PLOTS"
EXPLAIN_ARTIFACTS_DIR_ENV = "EXPLAIN_ARTIFACTS_DIR"
EXPLAIN_ENABLE_MALALA_XAI_ENV = "EXPLAIN_ENABLE_MALALA_XAI"
EXPLAIN_MALALA_RUN_SHAP_ENV = "EXPLAIN_MALALA_RUN_SHAP"
EXPLAIN_MALALA_RUN_ATTENTION_ENV = "EXPLAIN_MALALA_RUN_ATTENTION"
EXPLAIN_MALALA_RUN_GRADCAM_ENV = "EXPLAIN_MALALA_RUN_GRADCAM"
EXPLAIN_TIFF_QUALITY_MODE_ENV = "EXPLAIN_TIFF_QUALITY_MODE"  # auto | always | off
_DEFAULT_ARTIFACTS_DIR = _REPO_ROOT / "outputs" / "mcp_explainability"


def _default_checkpoint_name() -> str:
    value = os.environ.get(DEFAULT_EXPLAIN_CHECKPOINT_ENV, "fold4_best.pt")
    value = (value or "").strip()
    return value or "fold4_best.pt"

def _find_checkpoints_dir() -> Optional[Path]:
    seen: set[str] = set()
    for p in _CKPT_CANDIDATES:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p and p.exists():
            pts = list(p.glob("fold*_best.pt"))
            if pts:
                return p
    return None

DEVICE = torch.device(os.environ.get("EXPLAIN_DEVICE", "cpu"))

ENV_DIR = DATA_DIR / "EnvironmentalValues"


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _artifact_root_dir() -> Path:
    raw = (os.environ.get(EXPLAIN_ARTIFACTS_DIR_ENV, "") or "").strip()
    return Path(raw).expanduser() if raw else _DEFAULT_ARTIFACTS_DIR


def _prepare_artifact_dir(tool_name: str, survey_id: int) -> Path:
    out_dir = _artifact_root_dir() / tool_name / f"survey_{int(survey_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


_MALALA_XAI_MODULE: Optional[Any] = None
_MALALA_XAI_MODULE_ERROR: Optional[str] = None


def _load_malala_xai_module() -> Tuple[Optional[Any], Optional[str]]:
    global _MALALA_XAI_MODULE, _MALALA_XAI_MODULE_ERROR
    if _MALALA_XAI_MODULE is not None:
        return _MALALA_XAI_MODULE, None
    if _MALALA_XAI_MODULE_ERROR is not None:
        return None, _MALALA_XAI_MODULE_ERROR

    module_path = _REPO_ROOT / "Malala" / "explainability.py"
    if not module_path.exists():
        _MALALA_XAI_MODULE_ERROR = f"Module not found: {module_path}"
        return None, _MALALA_XAI_MODULE_ERROR

    try:
        spec = importlib.util.spec_from_file_location("malala_explainability", module_path)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to create import spec.")
        module = importlib.util.module_from_spec(spec)
        # Important: MCP stdio transport cannot tolerate arbitrary stdout/stderr.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
        _MALALA_XAI_MODULE = module
        return module, None
    except Exception as exc:
        _MALALA_XAI_MODULE_ERROR = str(exc)
        return None, _MALALA_XAI_MODULE_ERROR


def _run_quietly(fn: Any, *args: Any, **kwargs: Any) -> Tuple[Any, str]:
    """
    Execute a function while capturing stdout/stderr to protect MCP JSON-RPC stdio.
    Returns (result, captured_log_text).
    """
    out = io.StringIO()
    err = io.StringIO()
    result: Any = None

    # Some third-party libs write directly to file descriptors 1/2
    # (bypassing redirect_stdout/redirect_stderr). We capture both layers.
    fd_logs: List[str] = []
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with tempfile.TemporaryFile(mode="w+b") as tmp_out, tempfile.TemporaryFile(mode="w+b") as tmp_err:
            os.dup2(tmp_out.fileno(), stdout_fd)
            os.dup2(tmp_err.fileno(), stderr_fd)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                result = fn(*args, **kwargs)

            tmp_out.flush()
            tmp_err.flush()
            tmp_out.seek(0)
            tmp_err.seek(0)
            raw_out = tmp_out.read().decode("utf-8", errors="replace").strip()
            raw_err = tmp_err.read().decode("utf-8", errors="replace").strip()
            if raw_out:
                fd_logs.append(raw_out)
            if raw_err:
                fd_logs.append(raw_err)
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

    logs = "\n".join(
        part for part in [out.getvalue().strip(), err.getvalue().strip(), *fd_logs] if part
    )
    return result, logs

# ──────────────────────────────────────────────────────────────
#  HELPER ERREUR (même convention que tools_data / tools_stats)
# ──────────────────────────────────────────────────────────────

def _error(error: str, hint: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "error": error, "hint": hint}
    payload.update(extra)
    return payload

# ──────────────────────────────────────────────────────────────
#  IMPORT DU MODÈLE
#  On essaie d'importer depuis le dossier courant ou le parent
# ──────────────────────────────────────────────────────────────

def _import_model_class():
    """Importe MultimodalPlantModel depuis model.py (même dossier ou parent)."""
    search = [
        Path(__file__).parent,
        Path(__file__).parent.parent,
        Path("."),
    ]
    for p in search:
        model_py = p / "model.py"
        if model_py.exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            from model import MultimodalPlantModel
            return MultimodalPlantModel
    raise ImportError(
        "model.py introuvable. Placer tools_explain.py dans le même "
        "dossier que model.py ou définir le chemin dans sys.path."
    )

# ──────────────────────────────────────────────────────────────
#  CHARGEMENT DU CHECKPOINT
# ──────────────────────────────────────────────────────────────

def _load_checkpoint(checkpoint_name: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
    """
    Charge un checkpoint et reconstruit le modèle.
    Retourne (model, checkpoint_meta).
    """
    requested_checkpoint = (checkpoint_name or "").strip() or _default_checkpoint_name()
    ckpt_dir = _find_checkpoints_dir()
    if ckpt_dir is None:
        searched = [str(p) for p in _CKPT_CANDIDATES]
        raise FileNotFoundError(
            "Aucun dossier de checkpoints trouvé. "
            f"Définir CHECKPOINTS_DIR ou vérifier les chemins: {searched}"
        )

    ckpt_path = ckpt_dir / requested_checkpoint
    if not ckpt_path.exists():
        # Essayer de trouver un checkpoint correspondant ; sinon fallback propre.
        candidates = list(ckpt_dir.glob("fold*_best.pt"))
        if not candidates:
            raise FileNotFoundError(f"Aucun checkpoint dans {ckpt_dir}")
        default_candidate = ckpt_dir / _default_checkpoint_name()
        if default_candidate.exists():
            ckpt_path = default_candidate
        else:
            ckpt_path = sorted(candidates)[0]

    MultimodalPlantModel = _import_model_class()
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)

    model = MultimodalPlantModel(
        n_species        = ckpt["n_classes"],
        env_dim          = ckpt["env_dim"],
        aux_dim          = ckpt["aux_dim"],
        landsat_features = 24,
        climate_features = 76,
        branch_dim       = ckpt.get("branch_dim", 128),
        fusion_hidden_dim= ckpt.get("fusion_hidden_dim", 512),
        dropout          = ckpt.get("dropout", 0.2),
        use_image_branch = ckpt.get("use_image_branch", False),
    ).to(DEVICE)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_requested": requested_checkpoint,
        "fold":       ckpt.get("fold"),
        "epoch":      ckpt.get("epoch"),
        "score":      ckpt.get("score"),
        "alpha":      ckpt.get("alpha"),
        "n_classes":  ckpt["n_classes"],
        "env_dim":    ckpt["env_dim"],
        "aux_dim":    ckpt["aux_dim"],
    }

# ──────────────────────────────────────────────────────────────
#  PRÉPARATION DES DONNÉES POUR UN SURVEY
# ──────────────────────────────────────────────────────────────

# Noms des features env dans le même ordre que train.py
# (bioclim + soilgrids + elevation + landcover + human_footprint)
_BIO_COLS  = [f"Bio{i}" for i in range(1, 20)]
_SOIL_COLS = None   # détectés dynamiquement
_ELEV_COL  = "Elevation"

_ENV_CACHE: Dict[str, Any] = {}

def _load_env_tables() -> pd.DataFrame:
    """Charge et merge les CSV environnementaux (mis en cache)."""
    if "merged" in _ENV_CACHE:
        return _ENV_CACHE["merged"]

    dfs = []
    csvs = {
        "bioclim": ENV_DIR / "ClimateAverage_1981-2010" / "GLC25-PA-train-bioclimatic.csv",
        "soil":    ENV_DIR / "SoilGrids"                / "GLC25-PA-train-soilgrids.csv",
        "elev":    ENV_DIR / "Elevation"                / "GLC25-PA-train-elevation.csv",
        "lc":      ENV_DIR / "LandCover"                / "GLC25-PA-train-landcover.csv",
        "hfp":     ENV_DIR / "HumanFootprint"           / "GLC25-PA-train-human_footprint.csv",
    }
    for name, path in csvs.items():
        if path.exists():
            df = pd.read_csv(path)
            df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
            df["surveyId"] = df["surveyId"].astype(int)
            dfs.append(df.set_index("surveyId"))

    if not dfs:
        raise FileNotFoundError("Aucun CSV environnemental trouvé dans ENV_DIR.")

    merged = dfs[0].join(dfs[1:], how="outer")
    _ENV_CACHE["merged"] = merged
    _ENV_CACHE["feature_names"] = merged.columns.tolist()
    return merged


def _load_meta_table() -> pd.DataFrame:
    """Charge les métadonnées PA (une ligne par surveyId unique)."""
    if "meta" in _ENV_CACHE:
        return _ENV_CACHE["meta"]
    path = DATA_DIR / "GLC25_PA_metadata_train.csv"
    df = pd.read_csv(path)
    df["surveyId"] = df["surveyId"].astype(int)
    # One-hot region + country comme dans train.py
    df_u = df.drop_duplicates("surveyId").set_index("surveyId")
    _ENV_CACHE["meta"] = df_u
    return df_u


def _find_pt_file(directory: Path, survey_id: int) -> Optional[Path]:
    """Cherche un fichier .pt correspondant à un surveyId."""
    if not directory.exists():
        return None
    sid_str = str(survey_id)
    # Chercher par glob récursif (peut être lent - mis en cache par train.py)
    for f in directory.rglob("*.pt"):
        for token in reversed(f.stem.split("_")):
            if token.isdigit() and int(token) == survey_id:
                return f
    return None


def _normalize_series(t: torch.Tensor) -> torch.Tensor:
    """Normalisation z-score par instance (robust aux NaN)."""
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).float()
    std = t.std()
    if std < 1e-6 or torch.isnan(std):
        std = torch.tensor(1.0)
    return (t - t.mean()) / std


def _get_survey_tensors(
    survey_id: int,
    env_dim: int,
    aux_dim: int,
) -> Dict[str, Any]:
    """
    Construit les 4 tenseurs d'entrée pour le modèle.
    Retourne dict avec : x_env, x_aux, x_landsat, x_climate, feature_names
    """
    env_table  = _load_env_tables()
    meta_table = _load_meta_table()

    # ── Features environnementales tabulaires ──────────────────
    if survey_id not in env_table.index:
        raise ValueError(f"survey_id={survey_id} absent du CSV environnemental.")

    env_row = env_table.loc[survey_id]
    feature_names = env_table.columns.tolist()

    # Normalisation z-score sur tout le dataset (approximation)
    env_means = env_table.mean()
    env_stds  = env_table.std().replace(0, 1)
    env_norm  = (env_row - env_means) / env_stds
    env_raw_values = env_row.values.astype(np.float32)
    env_norm  = env_norm.fillna(0).values.astype(np.float32)

    # Adapter la dimension si env_dim != nombre de colonnes disponibles
    # (train.py peut avoir plus ou moins de colonnes selon le run)
    if len(env_norm) > env_dim:
        env_norm = env_norm[:env_dim]
    elif len(env_norm) < env_dim:
        env_norm = np.pad(env_norm, (0, env_dim - len(env_norm)))
    feature_names = feature_names[:env_dim]
    env_raw_values = env_raw_values[:env_dim] if len(env_raw_values) >= env_dim else np.pad(env_raw_values, (0, env_dim - len(env_raw_values)), constant_values=np.nan)

    x_env = torch.tensor(env_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ── Features auxiliaires (GPS + région + pays) ─────────────
    # Approximation : lon, lat, year, geoUncertainty, areaInM2 + one-hot
    aux_raw = np.zeros(aux_dim, dtype=np.float32)
    aux_raw_values = np.full(aux_dim, np.nan, dtype=np.float32)
    aux_feature_names = [
        "aux_lon",
        "aux_lat",
        "aux_year",
        "aux_geoUncertaintyInM",
        "aux_areaInM2",
    ]
    while len(aux_feature_names) < aux_dim:
        aux_feature_names.append(f"aux_extra_{len(aux_feature_names)}")
    if survey_id in meta_table.index:
        row = meta_table.loc[survey_id]
        # Positions 0-4 : lon, lat, year, geoUncertainty, areaInM2
        for i, col in enumerate(["lon", "lat", "year", "geoUncertaintyInM", "areaInM2"]):
            if col in meta_table.columns and i < aux_dim:
                v = row.get(col, 0)
                if pd.notna(v):
                    aux_raw[i] = float(v)
                    aux_raw_values[i] = float(v)
                else:
                    aux_raw[i] = 0.0
    x_aux = torch.tensor(aux_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ── Landsat time series ────────────────────────────────────
    raw_landsat = np.full((6, 4, 21), np.nan, dtype=np.float32)
    landsat_dir = DATA_DIR / "SateliteTimeSeries-Landsat" / "cubes" / "PA-train"
    pt_file = _find_pt_file(landsat_dir, survey_id)
    if pt_file is not None:
        try:
            t = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            if isinstance(t, dict):
                t = next((v for v in t.values() if isinstance(v, torch.Tensor)), None)
            if t is not None:
                raw_candidate = t.detach().cpu().numpy().astype(np.float32)
                if raw_candidate.ndim == 3 and tuple(raw_candidate.shape) == (6, 4, 21):
                    raw_landsat = raw_candidate
                t = _normalize_series(t.float())
                # Reshape en [seq_len, features] = [21, 24]
                if t.ndim == 3:  # [6, 4, 21]
                    t = t.permute(2, 0, 1).reshape(21, 24)
                elif t.ndim == 2 and t.shape[0] == 21:
                    pass  # déjà [21, 24]
                x_landsat = t.unsqueeze(0).to(DEVICE)
            else:
                x_landsat = torch.zeros(1, 21, 24, device=DEVICE)
        except Exception:
            x_landsat = torch.zeros(1, 21, 24, device=DEVICE)
    else:
        x_landsat = torch.zeros(1, 21, 24, device=DEVICE)

    # ── Bioclim time series ─────────────────────────────────────
    raw_climate = np.full((4, 19, 12), np.nan, dtype=np.float32)
    bioclim_dir = DATA_DIR / "BioclimTimeSeries" / "cubes" / "PA-train"
    pt_file = _find_pt_file(bioclim_dir, survey_id)
    if pt_file is not None:
        try:
            t = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            if isinstance(t, dict):
                t = next((v for v in t.values() if isinstance(v, torch.Tensor)), None)
            if t is not None:
                raw_candidate = t.detach().cpu().numpy().astype(np.float32)
                if raw_candidate.ndim == 3 and tuple(raw_candidate.shape) == (4, 19, 12):
                    raw_climate = raw_candidate
                t = _normalize_series(t.float())
                # Reshape en [12, 76]
                if t.ndim == 3:  # [4, 19, 12]
                    t = t.permute(2, 0, 1).reshape(12, 76)
                elif t.ndim == 2 and t.shape[0] == 12:
                    pass
                x_climate = t.unsqueeze(0).to(DEVICE)
            else:
                x_climate = torch.zeros(1, 12, 76, device=DEVICE)
        except Exception:
            x_climate = torch.zeros(1, 12, 76, device=DEVICE)
    else:
        x_climate = torch.zeros(1, 12, 76, device=DEVICE)

    return {
        "x_env":      x_env,
        "x_aux":      x_aux,
        "x_landsat":  x_landsat,
        "x_climate":  x_climate,
        "x_env_raw": env_raw_values.astype(np.float32),
        "x_aux_raw": aux_raw_values.astype(np.float32),
        "aux_feature_names": aux_feature_names,
        "x_landsat_raw": raw_landsat.astype(np.float32),
        "x_climate_raw": raw_climate.astype(np.float32),
        "feature_names": feature_names,
    }

# ──────────────────────────────────────────────────────────────
#  ATTRIBUTION PAR GRADIENT × INPUT
# ──────────────────────────────────────────────────────────────

_LANDSAT_SEASON_LABELS = [
    f"{yr}-{q}"
    for yr in range(2000, 2021)
    for q in ["Hiver", "Print.", "Été", "Automne"]
][:21]  # 21 saisons disponibles

_BIOCLIM_MONTH_LABELS = [
    "Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
    "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"
]


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _feature_family_key(name: str) -> str:
    token = (name or "").lower()
    if token.startswith("bio"):
        return "bioclim"
    if token.startswith("soil") or token.startswith("phh2o") or token.startswith("bdod") or token.startswith("soc"):
        return "soil"
    if "elev" in token or "altitude" in token:
        return "elev"
    if "landcover" in token or "land_cover" in token:
        return "landcover"
    if "human" in token or "footprint" in token:
        return "humanfootprint"
    if token.startswith("aux_"):
        return "aux"
    return "other"


def _time_series_missing_profile(arr: np.ndarray, time_axis: int) -> Tuple[float, List[float], float]:
    safe = np.asarray(arr, dtype=np.float32)
    if safe.size == 0:
        return 1.0, [], 0.0

    total_missing = float(np.isnan(safe).sum() / safe.size)
    n_steps = safe.shape[time_axis]
    by_step: List[float] = []
    valid_steps = 0
    for i in range(n_steps):
        step = np.take(safe, i, axis=time_axis)
        missing_ratio = float(np.isnan(step).sum() / max(step.size, 1))
        by_step.append(round(missing_ratio, 4))
        if missing_ratio < 1.0:
            valid_steps += 1
    coverage = float(valid_steps / max(n_steps, 1))
    return round(total_missing, 4), by_step, round(coverage, 4)


def _extract_importance_vector(
    full_items: Any,
    value_key: str,
    label_key: str,
) -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    values: List[float] = []
    if isinstance(full_items, list):
        for item in full_items:
            if not isinstance(item, dict):
                continue
            labels.append(str(item.get(label_key, "")))
            try:
                values.append(float(item.get(value_key, 0.0)))
            except Exception:
                values.append(0.0)
    arr = np.asarray(values, dtype=np.float32) if values else np.asarray([], dtype=np.float32)
    return labels, arr


def _mean_on_indices(values: List[float], indices: np.ndarray) -> float:
    if not values or indices.size == 0:
        return 0.0
    picked = [float(values[i]) for i in indices.tolist() if 0 <= int(i) < len(values)]
    if not picked:
        return 0.0
    return float(round(float(np.mean(picked)), 4))


def _collect_salient_feature_names(explanations: Any, top_n: int = 12) -> List[str]:
    scored: List[Tuple[str, float]] = []
    if not isinstance(explanations, list):
        return []
    for exp in explanations:
        if not isinstance(exp, dict):
            continue
        for feat in exp.get("top_env_features", []) or []:
            if not isinstance(feat, dict):
                continue
            name = str(feat.get("feature", "")).strip()
            if not name:
                continue
            try:
                score = float(feat.get("abs_score", abs(float(feat.get("attribution", 0.0)))))
            except Exception:
                score = 0.0
            scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    out: List[str] = []
    seen: set[str] = set()
    for name, _ in scored:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= top_n:
            break
    return out


def _build_tabular_reference_stats() -> Dict[str, Any]:
    global _TABULAR_REF_CACHE
    if _TABULAR_REF_CACHE is not None:
        return _TABULAR_REF_CACHE

    env_table = _load_env_tables()
    meta_table = _load_meta_table()
    stats: Dict[str, Dict[str, float]] = {}

    def _record(series: pd.Series, feature_name: str) -> None:
        numeric = pd.to_numeric(series, errors="coerce")
        q25 = float(numeric.quantile(0.25)) if numeric.notna().any() else 0.0
        q75 = float(numeric.quantile(0.75)) if numeric.notna().any() else 0.0
        iqr = float(q75 - q25)
        mean = float(numeric.mean()) if numeric.notna().any() else 0.0
        std = float(numeric.std()) if numeric.notna().any() else 0.0
        med = float(numeric.median()) if numeric.notna().any() else 0.0
        stats[feature_name] = {
            "median": med,
            "iqr": iqr,
            "mean": mean,
            "std": std,
            "family": _feature_family_key(feature_name),
        }

    for col in env_table.columns:
        _record(env_table[col], str(col))

    aux_cols = {
        "aux_lon": "lon",
        "aux_lat": "lat",
        "aux_year": "year",
        "aux_geoUncertaintyInM": "geoUncertaintyInM",
        "aux_areaInM2": "areaInM2",
    }
    for out_name, src_col in aux_cols.items():
        if src_col in meta_table.columns:
            _record(meta_table[src_col], out_name)
        else:
            stats[out_name] = {
                "median": 0.0,
                "iqr": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "family": "aux",
            }

    _TABULAR_REF_CACHE = {"feature_stats": stats}
    return _TABULAR_REF_CACHE


def _compute_tabular_quality(
    tensors: Dict[str, Any],
    salient_feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ref = _build_tabular_reference_stats()["feature_stats"]
    env_names = list(tensors.get("feature_names", []))
    env_raw = np.asarray(tensors.get("x_env_raw", []), dtype=np.float32)
    aux_names = list(tensors.get("aux_feature_names", []))
    aux_raw = np.asarray(tensors.get("x_aux_raw", []), dtype=np.float32)

    values: Dict[str, float] = {}
    for i, name in enumerate(env_names):
        values[name] = float(env_raw[i]) if i < len(env_raw) else float("nan")
    for i, name in enumerate(aux_names):
        values[name] = float(aux_raw[i]) if i < len(aux_raw) else float("nan")

    total = max(len(values), 1)
    missing_features = [k for k, v in values.items() if np.isnan(v)]
    missing_ratio = float(len(missing_features) / total)

    extremes: List[Dict[str, Any]] = []
    low_quality_names: set[str] = set(missing_features)
    family_bins: Dict[str, Dict[str, float]] = {
        key: {"n_features": 0.0, "missing_count": 0.0, "extreme_count": 0.0}
        for key in ["bioclim", "soil", "elev", "landcover", "humanfootprint", "aux", "other"]
    }

    for feat, val in values.items():
        fam = _feature_family_key(feat)
        family_bins[fam]["n_features"] += 1
        if np.isnan(val):
            family_bins[fam]["missing_count"] += 1
            continue

        st = ref.get(feat, {"median": 0.0, "iqr": 0.0, "std": 0.0})
        med = float(st.get("median", 0.0))
        iqr = float(st.get("iqr", 0.0))
        std = float(st.get("std", 0.0))
        scale = iqr / 1.349 if iqr > 1e-6 else std
        if scale <= 1e-6:
            z_robust = 0.0
        else:
            z_robust = float((val - med) / scale)
        if abs(z_robust) > 3.0:
            family_bins[fam]["extreme_count"] += 1
            low_quality_names.add(feat)
            extremes.append(
                {
                    "feature": feat,
                    "value": round(float(val), 4),
                    "z_robust": round(float(z_robust), 3),
                    "family": fam,
                }
            )

    extremes.sort(key=lambda x: abs(float(x["z_robust"])), reverse=True)
    extreme_ratio = float(len(extremes) / max(total - len(missing_features), 1))

    salient = [f for f in (salient_feature_names or []) if isinstance(f, str) and f]
    salient_low = sorted([f for f in salient if f in low_quality_names])
    salient_low_count = len(salient_low)
    salient_penalty = float(salient_low_count / max(len(salient), 1)) if salient else 0.0

    family_quality: Dict[str, Any] = {}
    for fam, counters in family_bins.items():
        n = max(int(counters["n_features"]), 1)
        missing_r = float(counters["missing_count"] / n)
        extreme_r = float(counters["extreme_count"] / n)
        low_r = _clamp01(0.6 * missing_r + 0.4 * extreme_r)
        family_quality[fam] = {
            "n_features": int(counters["n_features"]),
            "missing_ratio": round(missing_r, 4),
            "extreme_ratio": round(extreme_r, 4),
            "low_quality_ratio": round(low_r, 4),
            "quality_score": round(1.0 - low_r, 4),
        }

    tabular_penalty = _clamp01(0.5 * missing_ratio + 0.3 * extreme_ratio + 0.2 * salient_penalty)

    return {
        "tabular_missing_ratio": round(missing_ratio, 4),
        "imputed_feature_count": len(missing_features),
        "imputed_features_top": missing_features[:12],
        "extreme_features": extremes[:20],
        "family_quality": family_quality,
        "salient_tabular_features_low_quality_count": salient_low_count,
        "salient_tabular_features_low_quality": salient_low,
        "flags": {
            "tabular_many_missing": missing_ratio > 0.10,
            "tabular_many_extremes": len(extremes) >= 5,
            "salient_features_unreliable": salient_low_count >= 2,
        },
        "tabular_penalty": round(tabular_penalty, 4),
    }


def _compute_time_series_quality(
    tensors: Dict[str, Any],
    temporal_importance: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_landsat = np.asarray(tensors.get("x_landsat_raw", np.full((6, 4, 21), np.nan, dtype=np.float32)))
    raw_bioclim = np.asarray(tensors.get("x_climate_raw", np.full((4, 19, 12), np.nan, dtype=np.float32)))

    l_total, l_by_step, l_cov = _time_series_missing_profile(raw_landsat, time_axis=2)
    b_total, b_by_month, b_cov = _time_series_missing_profile(raw_bioclim, time_axis=2)

    seasonal_full = []
    monthly_full = []
    if isinstance(temporal_importance, dict):
        seasonal_full = temporal_importance.get("landsat_seasonal_importance_full") or []
        monthly_full = temporal_importance.get("bioclim_monthly_importance_full") or []

    l_labels, l_imp_pct = _extract_importance_vector(seasonal_full, "importance_pct", "season")
    b_labels, b_imp_pct = _extract_importance_vector(monthly_full, "importance_pct", "month")

    if l_imp_pct.size == 0:
        l_imp_pct = np.ones(len(l_by_step), dtype=np.float32) * (100.0 / max(len(l_by_step), 1))
        l_labels = _LANDSAT_SEASON_LABELS[: len(l_by_step)]
    if b_imp_pct.size == 0:
        b_imp_pct = np.ones(len(b_by_month), dtype=np.float32) * (100.0 / max(len(b_by_month), 1))
        b_labels = _BIOCLIM_MONTH_LABELS[: len(b_by_month)]

    l_top_idx = np.argsort(l_imp_pct)[::-1][:5]
    b_top_idx = np.argsort(b_imp_pct)[::-1][:5]
    salient_timestep_missing = _mean_on_indices(l_by_step, l_top_idx)
    salient_month_missing = _mean_on_indices(b_by_month, b_top_idx)

    winter_l = float(sum(v for lab, v in zip(l_labels, l_imp_pct.tolist()) if "hiver" in str(lab).lower()))
    summer_l = float(sum(v for lab, v in zip(l_labels, l_imp_pct.tolist()) if "été" in str(lab).lower() or "ete" in str(lab).lower()))

    month_name_to_idx = {name.lower(): i for i, name in enumerate(_BIOCLIM_MONTH_LABELS)}
    winter_idx = {11, 0, 1}
    summer_idx = {5, 6, 7}
    winter_b = 0.0
    summer_b = 0.0
    for lab, val in zip(b_labels, b_imp_pct.tolist()):
        idx = month_name_to_idx.get(str(lab).lower())
        if idx in winter_idx:
            winter_b += float(val)
        if idx in summer_idx:
            summer_b += float(val)

    coverage_penalty = _clamp01((max(0.0, 0.70 - l_cov) / 0.70 + max(0.0, 0.70 - b_cov) / 0.70) / 2.0)

    return {
        "landsat_missing_ratio_total": l_total,
        "landsat_missing_ratio_by_timestep": l_by_step,
        "landsat_temporal_coverage": l_cov,
        "bioclim_missing_ratio_total": b_total,
        "bioclim_missing_ratio_by_month": b_by_month,
        "bioclim_temporal_coverage": b_cov,
        "salient_timesteps_missing_ratio": salient_timestep_missing,
        "salient_months_missing_ratio": salient_month_missing,
        "winter_importance_pct": round(winter_l, 2),
        "summer_importance_pct": round(summer_l, 2),
        "bioclim_winter_months_pct": round(winter_b, 2),
        "bioclim_summer_months_pct": round(summer_b, 2),
        "flags": {
            "salient_periods_degraded": salient_timestep_missing >= 0.25 or salient_month_missing >= 0.25,
            "high_overall_ts_missingness": ((l_total + b_total) / 2.0) >= 0.20,
            "low_temporal_coverage": l_cov < 0.70 or b_cov < 0.70,
            "winter_dominant_signal": (winter_l / 100.0) >= 0.35,
        },
        "overall_ts_missingness": round((l_total + b_total) / 2.0, 4),
        "coverage_penalty": round(coverage_penalty, 4),
    }


def _quality_confidence_from_components(
    time_series_quality: Dict[str, Any],
    tabular_quality: Dict[str, Any],
) -> Dict[str, Any]:
    salient_ts_missing = float(
        (
            float(time_series_quality.get("salient_timesteps_missing_ratio", 0.0))
            + float(time_series_quality.get("salient_months_missing_ratio", 0.0))
        )
        / 2.0
    )
    overall_ts_missing = float(time_series_quality.get("overall_ts_missingness", 0.0))
    coverage_penalty = float(time_series_quality.get("coverage_penalty", 0.0))
    tabular_penalty = float(tabular_quality.get("tabular_penalty", 0.0))

    quality_confidence = _clamp01(
        1.0
        - (
            0.35 * salient_ts_missing
            + 0.20 * overall_ts_missing
            + 0.20 * coverage_penalty
            + 0.25 * tabular_penalty
        )
    )

    if quality_confidence >= 0.75:
        level = "high"
    elif quality_confidence >= 0.50:
        level = "medium"
    else:
        level = "low"

    return {
        "quality_confidence": round(quality_confidence, 4),
        "confidence_level": level,
        "components": {
            "salient_ts_missing": round(salient_ts_missing, 4),
            "overall_ts_missing": round(overall_ts_missing, 4),
            "coverage_penalty": round(coverage_penalty, 4),
            "tabular_penalty": round(tabular_penalty, 4),
        },
    }


def _gradient_x_input_attribution(
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
    top_species_indices: List[int],
) -> Dict[str, Any]:
    """
    Gradient × Input pour les features tabulaires env.
    Pour chaque espèce prédite : quelle variable environnementale a le plus compté.
    """
    x_env     = tensors["x_env"].clone().requires_grad_(True)
    x_aux     = tensors["x_aux"].clone().requires_grad_(True)
    x_landsat = tensors["x_landsat"].clone().requires_grad_(True)
    x_climate = tensors["x_climate"].clone().requires_grad_(True)
    feat_names = tensors["feature_names"]

    results = []
    for sp_idx in top_species_indices:
        model.zero_grad()
        logits = model(x_env, x_aux, x_landsat, x_climate)
        # Gradient du score de cette espèce w.r.t. toutes les entrées
        score = logits[0, sp_idx]
        score.backward(retain_graph=True)

        # Gradient × input pour env features
        grad_env = x_env.grad.detach().cpu().squeeze(0).numpy()
        inp_env  = x_env.detach().cpu().squeeze(0).numpy()
        attr_env = (grad_env * inp_env)  # [env_dim]

        # Regrouper par famille de features
        families = _group_by_family(attr_env, feat_names)

        # Top 5 features individuelles
        abs_attr = np.abs(attr_env)
        top_indices = np.argsort(abs_attr)[::-1][:8]
        top_features = [
            {
                "feature":     feat_names[i] if i < len(feat_names) else f"feat_{i}",
                "attribution": float(round(attr_env[i], 4)),
                "abs_score":   float(round(abs_attr[i], 4)),
                "direction":   "positive" if attr_env[i] > 0 else "negative",
            }
            for i in top_indices
        ]

        # Gradient sur Landsat : importance par saison
        grad_land = x_landsat.grad.detach().cpu().squeeze(0).numpy()  # [21, 24]
        season_importance = np.abs(grad_land).mean(axis=1)  # [21]
        season_importance = season_importance / (season_importance.sum() + 1e-8)
        top_seasons_idx = np.argsort(season_importance)[::-1][:5]
        top_seasons = [
            {
                "season": _LANDSAT_SEASON_LABELS[i] if i < len(_LANDSAT_SEASON_LABELS) else f"saison_{i}",
                "importance_pct": float(round(season_importance[i] * 100, 2)),
            }
            for i in top_seasons_idx
        ]

        # Gradient sur Bioclim : importance par mois
        grad_clim = x_climate.grad.detach().cpu().squeeze(0).numpy()  # [12, 76]
        month_importance = np.abs(grad_clim).mean(axis=1)  # [12]
        month_importance = month_importance / (month_importance.sum() + 1e-8)
        top_months_idx = np.argsort(month_importance)[::-1][:5]
        top_months = [
            {
                "month":          _BIOCLIM_MONTH_LABELS[i] if i < len(_BIOCLIM_MONTH_LABELS) else f"mois_{i}",
                "importance_pct": float(round(month_importance[i] * 100, 2)),
            }
            for i in top_months_idx
        ]

        results.append({
            "species_idx":       sp_idx,
            "top_env_features":  top_features,
            "feature_families":  families,
            "top_landsat_seasons": top_seasons,
            "top_bioclim_months":  top_months,
        })

        # Réinitialiser les gradients pour la prochaine espèce
        if x_env.grad is not None:
            x_env.grad.zero_()
        if x_aux.grad is not None:
            x_aux.grad.zero_()
        if x_landsat.grad is not None:
            x_landsat.grad.zero_()
        if x_climate.grad is not None:
            x_climate.grad.zero_()

    return results


def _group_by_family(
    attributions: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Regroupe les attributions par famille de variables."""
    families: Dict[str, float] = {
        "BioClim":       0.0,
        "Sol":           0.0,
        "Élévation":     0.0,
        "LandCover":     0.0,
        "Human footprint": 0.0,
        "Autre":         0.0,
    }
    for i, name in enumerate(feature_names):
        if i >= len(attributions):
            break
        v = float(abs(attributions[i]))
        n = name.lower()
        if n.startswith("bio"):
            families["BioClim"] += v
        elif "soil" in n or n.startswith("phh2o") or n.startswith("bdod") or n.startswith("soc"):
            families["Sol"] += v
        elif "elev" in n or "altitude" in n:
            families["Élévation"] += v
        elif "landcover" in n or "land_cover" in n:
            families["LandCover"] += v
        elif "human" in n or "footprint" in n:
            families["Human footprint"] += v
        else:
            families["Autre"] += v

    total = sum(families.values()) + 1e-8
    return {k: float(round(v / total * 100, 2)) for k, v in families.items()}

# ──────────────────────────────────────────────────────────────
#  IMPORTANCE DES BRANCHES PAR NORME L2
# ──────────────────────────────────────────────────────────────

def _branch_importance(
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Norme L2 de l'embedding de chaque branche.
    Indique quelle branche contribue le plus à la décision finale.
    """
    with torch.no_grad():
        emb_env     = model.environment_encoder(tensors["x_env"])     # [1, 128]
        emb_aux     = model.aux_encoder(tensors["x_aux"])             # [1, 128]
        emb_landsat = model.landsat_encoder(tensors["x_landsat"])     # [1, 128]
        emb_climate = model.climate_encoder(tensors["x_climate"])     # [1, 128]

    norms = {
        "Env. tabulaires (MLP)":         float(emb_env.norm().item()),
        "GPS + région (MLP)":            float(emb_aux.norm().item()),
        "Landsat séries temp. (Transf.)":float(emb_landsat.norm().item()),
        "Bioclim séries temp. (Transf.)":float(emb_climate.norm().item()),
    }
    total = sum(norms.values()) + 1e-8
    return {k: float(round(v / total * 100, 2)) for k, v in norms.items()}

# ──────────────────────────────────────────────────────────────
#  POIDS D'ATTENTION DU TRANSFORMER (HOOK)
# ──────────────────────────────────────────────────────────────

def _extract_attention_weights(
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Capture les poids d'attention des TransformerEncoderLayer via hooks.
    Retourne l'importance de chaque timestep pour Landsat et Bioclim.
    """
    attn_weights: Dict[str, List[torch.Tensor]] = {
        "landsat": [],
        "climate": [],
    }

    def _make_hook(key: str):
        def hook(module, input, output):
            # PyTorch TransformerEncoderLayer : on patche temporairement
            # pour récupérer les poids d'attention via forward du sous-module
            pass
        return hook

    # Approche plus simple et robuste : gradient de la moyenne sur les timesteps
    x_l = tensors["x_landsat"].clone().requires_grad_(True)
    x_c = tensors["x_climate"].clone().requires_grad_(True)

    with torch.enable_grad():
        # Forward complet
        emb_l = model.landsat_encoder(x_l)   # [1, 128]
        emb_c = model.climate_encoder(x_c)   # [1, 128]

        # Gradient w.r.t. à chaque timestep Landsat
        loss_l = emb_l.sum()
        loss_l.backward()
        if x_l.grad is not None:
            grad_l = x_l.grad.detach().cpu().squeeze(0).numpy()  # [21, 24]
            season_imp = np.abs(grad_l).mean(axis=1)
            season_imp = season_imp / (season_imp.sum() + 1e-8)
        else:
            season_imp = np.ones(21) / 21

        # Gradient w.r.t. à chaque mois Bioclim
        loss_c = emb_c.sum()
        loss_c.backward()
        if x_c.grad is not None:
            grad_c = x_c.grad.detach().cpu().squeeze(0).numpy()  # [12, 76]
            month_imp = np.abs(grad_c).mean(axis=1)
            month_imp = month_imp / (month_imp.sum() + 1e-8)
        else:
            month_imp = np.ones(12) / 12

    top_seasons_idx = np.argsort(season_imp)[::-1][:5]
    top_months_idx  = np.argsort(month_imp)[::-1][:5]

    landsat_full = [
        {
            "season": _LANDSAT_SEASON_LABELS[i] if i < len(_LANDSAT_SEASON_LABELS) else f"saison_{i}",
            "importance_pct": float(round(season_imp[i] * 100, 2)),
        }
        for i in range(len(season_imp))
    ]
    bioclim_full = [
        {
            "month": _BIOCLIM_MONTH_LABELS[i] if i < len(_BIOCLIM_MONTH_LABELS) else f"mois_{i}",
            "importance_pct": float(round(month_imp[i] * 100, 2)),
        }
        for i in range(len(month_imp))
    ]

    return {
        "landsat_seasonal_importance": [
            {
                "season":         _LANDSAT_SEASON_LABELS[i] if i < len(_LANDSAT_SEASON_LABELS) else f"saison_{i}",
                "importance_pct": float(round(season_imp[i] * 100, 2)),
            }
            for i in top_seasons_idx
        ],
        "landsat_seasonal_importance_full": landsat_full,
        "bioclim_monthly_importance": [
            {
                "month":          _BIOCLIM_MONTH_LABELS[i],
                "importance_pct": float(round(month_imp[i] * 100, 2)),
            }
            for i in top_months_idx
        ],
        "bioclim_monthly_importance_full": bioclim_full,
    }


def _init_artifacts_payload(tool_name: str, survey_id: int) -> Dict[str, Any]:
    if not _env_bool(EXPLAIN_SAVE_PLOTS_ENV, True):
        return {
            "enabled": False,
            "reason": f"{EXPLAIN_SAVE_PLOTS_ENV}=false",
        }
    if not HAS_MATPLOTLIB:
        return {
            "enabled": False,
            "reason": "matplotlib unavailable (install matplotlib to enable plots).",
        }
    out_dir = _prepare_artifact_dir(tool_name, survey_id)
    return {
        "enabled": True,
        "output_dir": str(out_dir),
        "plots": {},
        "warnings": [],
    }


def _plot_branch_importance_png(branch_importance: Dict[str, float], output_path: Path) -> None:
    assert HAS_MATPLOTLIB and plt is not None
    labels = list(branch_importance.keys())
    values = [float(branch_importance[k]) for k in labels]
    if not labels:
        return
    order = np.argsort(values)
    labels_sorted = [labels[i] for i in order]
    values_sorted = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(labels_sorted, values_sorted, color="#2E86AB")
    ax.set_xlabel("Contribution (%)")
    ax.set_title("Importance des branches du modèle")
    for bar, v in zip(bars, values_sorted):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2.0, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(values_sorted) * 1.2 if values_sorted else 100)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_temporal_importance_pngs(attn: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    assert HAS_MATPLOTLIB and plt is not None
    plots: Dict[str, str] = {}

    seasons = attn.get("landsat_seasonal_importance_full") or attn.get("landsat_seasonal_importance") or []
    if isinstance(seasons, list) and seasons:
        labels = [str(x.get("season", "")) for x in seasons if isinstance(x, dict)]
        values = [float(x.get("importance_pct", 0.0)) for x in seasons if isinstance(x, dict)]
        if labels and values:
            fig, ax = plt.subplots(figsize=(11, 4.6))
            ax.bar(range(len(values)), values, color="#1F77B4")
            ax.set_title("Importance temporelle Landsat")
            ax.set_ylabel("Importance (%)")
            step = 2 if len(labels) > 15 else 1
            ticks = list(range(0, len(labels), step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)
            plt.tight_layout()
            path = out_dir / "landsat_temporal_importance.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plots["landsat_temporal_importance_png"] = str(path)

    months = attn.get("bioclim_monthly_importance_full") or attn.get("bioclim_monthly_importance") or []
    if isinstance(months, list) and months:
        labels = [str(x.get("month", "")) for x in months if isinstance(x, dict)]
        values = [float(x.get("importance_pct", 0.0)) for x in months if isinstance(x, dict)]
        if labels and values:
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.bar(labels, values, color="#FF7F0E")
            ax.set_title("Importance mensuelle Bioclim")
            ax.set_ylabel("Importance (%)")
            plt.tight_layout()
            path = out_dir / "bioclim_monthly_importance.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plots["bioclim_monthly_importance_png"] = str(path)

    return plots


def _plot_species_env_attribution_pngs(
    explanations: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    assert HAS_MATPLOTLIB and plt is not None
    outputs: List[Dict[str, Any]] = []
    for exp in explanations:
        if not isinstance(exp, dict):
            continue
        top_feats = exp.get("top_env_features")
        if not isinstance(top_feats, list) or not top_feats:
            continue

        species_id = exp.get("species_id")
        species_idx = exp.get("species_idx")
        species_label = species_id if species_id is not None else species_idx
        token = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(species_label))
        if not token:
            token = "unknown"

        pairs: List[Tuple[str, float]] = []
        for item in top_feats:
            if not isinstance(item, dict):
                continue
            feature = str(item.get("feature", "feature"))
            try:
                value = float(item.get("attribution", 0.0))
            except Exception:
                value = 0.0
            pairs.append((feature, value))
        if not pairs:
            continue

        labels = [p[0] for p in pairs][::-1]
        values = [p[1] for p in pairs][::-1]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, values, color=colors)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(f"Gradient×Input env features - espèce {species_label}")
        ax.set_xlabel("Attribution (signed)")
        plt.tight_layout()
        path_attr = out_dir / f"species_{token}_env_attribution.png"
        fig.savefig(path_attr, dpi=150, bbox_inches="tight")
        plt.close(fig)

        item_out: Dict[str, Any] = {
            "species_id": species_id,
            "species_idx": species_idx,
            "env_attribution_png": str(path_attr),
        }

        families = exp.get("feature_families")
        if isinstance(families, dict) and families:
            fam_labels = list(families.keys())
            fam_values = [float(families[k]) for k in fam_labels]
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.bar(fam_labels, fam_values, color="#6C757D")
            ax.set_ylabel("Contribution (%)")
            ax.set_title(f"Importance par famille - espèce {species_label}")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            path_fam = out_dir / f"species_{token}_families.png"
            fig.savefig(path_fam, dpi=150, bbox_inches="tight")
            plt.close(fig)
            item_out["feature_families_png"] = str(path_fam)

        outputs.append(item_out)

    return outputs


def _plot_missingness_timeline_pngs(time_series_quality: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    assert HAS_MATPLOTLIB and plt is not None
    plots: Dict[str, str] = {}

    landsat_missing = time_series_quality.get("landsat_missing_ratio_by_timestep", [])
    if isinstance(landsat_missing, list) and landsat_missing:
        labels = _LANDSAT_SEASON_LABELS[: len(landsat_missing)]
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(range(len(landsat_missing)), landsat_missing, marker="o", linewidth=1.4, color="#D62728")
        ax.set_ylim(0, 1)
        ax.set_title("Missingness Landsat par saison")
        ax.set_ylabel("Ratio manquant")
        step = 2 if len(labels) > 15 else 1
        ticks = list(range(0, len(labels), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        path = out_dir / "missingness_timeline_landsat.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["missingness_timeline_landsat_png"] = str(path)

    bioclim_missing = time_series_quality.get("bioclim_missing_ratio_by_month", [])
    if isinstance(bioclim_missing, list) and bioclim_missing:
        labels = _BIOCLIM_MONTH_LABELS[: len(bioclim_missing)]
        fig, ax = plt.subplots(figsize=(8.8, 4.3))
        ax.bar(labels, bioclim_missing, color="#9467BD")
        ax.set_ylim(0, 1)
        ax.set_title("Missingness Bioclim par mois")
        ax.set_ylabel("Ratio manquant")
        plt.tight_layout()
        path = out_dir / "missingness_timeline_bioclim.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["missingness_timeline_bioclim_png"] = str(path)

    return plots


def _plot_tabular_quality_overview_png(tabular_quality: Dict[str, Any], output_path: Path) -> None:
    assert HAS_MATPLOTLIB and plt is not None
    families = tabular_quality.get("family_quality", {})
    if not isinstance(families, dict) or not families:
        return

    labels: List[str] = []
    values: List[float] = []
    for fam in ["bioclim", "soil", "elev", "landcover", "humanfootprint", "aux"]:
        if fam not in families:
            continue
        item = families.get(fam)
        if not isinstance(item, dict):
            continue
        labels.append(fam)
        values.append(float(item.get("low_quality_ratio", 0.0)))
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(labels, values, color="#6C757D")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Low-quality ratio")
    ax.set_title("Qualité des données tabulaires par famille")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_quality_explanation_summary(
    time_series_quality: Dict[str, Any],
    tabular_quality: Dict[str, Any],
    confidence_level: str,
    quality_confidence: float,
    tiff_quality: Optional[Dict[str, Any]] = None,
) -> str:
    if confidence_level == "high":
        intro = f"Interprétation probable (score confiance={quality_confidence:.2f})."
    elif confidence_level == "medium":
        intro = f"Interprétation possible avec réserve (score confiance={quality_confidence:.2f})."
    else:
        intro = f"Hypothèse fragile avec forte incertitude (score confiance={quality_confidence:.2f})."

    reasons: List[str] = []
    ts_flags = time_series_quality.get("flags", {}) if isinstance(time_series_quality.get("flags"), dict) else {}
    tab_flags = tabular_quality.get("flags", {}) if isinstance(tabular_quality.get("flags"), dict) else {}

    if ts_flags.get("salient_periods_degraded"):
        reasons.append(
            "Les périodes temporelles les plus influentes contiennent des données manquantes significatives."
        )
    if ts_flags.get("winter_dominant_signal"):
        reasons.append("Le signal utilisé par le modèle est dominé par des périodes hivernales.")
    if tab_flags.get("salient_features_unreliable"):
        reasons.append("Plusieurs variables tabulaires saillantes sont atypiques ou imputées.")
    if tab_flags.get("tabular_many_missing"):
        reasons.append("Une part non négligeable des variables tabulaires est manquante/imputée.")

    if isinstance(tiff_quality, dict) and tiff_quality.get("available"):
        haze = float(tiff_quality.get("haze_cloud_proxy_ratio", 0.0))
        if haze >= 0.20:
            reasons.append(f"Le proxy nuage/voile est élevé ({haze:.2f}), ce qui peut dégrader l'information image.")

    if not reasons:
        reasons.append("Aucun signal majeur de dégradation n'a été détecté sur les entrées principales.")

    return intro + " " + " ".join(reasons[:3])


def _compute_observation_quality(
    survey_id: int,
    tensors: Dict[str, Any],
    temporal_importance: Optional[Dict[str, Any]],
    explanations: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    salient_names = _collect_salient_feature_names(explanations or [], top_n=12)
    ts_quality = _compute_time_series_quality(tensors, temporal_importance)
    tab_quality = _compute_tabular_quality(tensors, salient_feature_names=salient_names)

    conf = _quality_confidence_from_components(ts_quality, tab_quality)
    level = str(conf["confidence_level"])

    mode = _resolve_tiff_quality_mode()
    run_tiff_detail = mode == "always" or (mode == "auto" and level != "high")
    tiff_quality: Optional[Dict[str, Any]] = None
    if run_tiff_detail:
        tiff_quality = _compute_tiff_quality(int(survey_id))

    summary = _build_quality_explanation_summary(
        time_series_quality=ts_quality,
        tabular_quality=tab_quality,
        confidence_level=level,
        quality_confidence=float(conf["quality_confidence"]),
        tiff_quality=tiff_quality,
    )

    payload: Dict[str, Any] = {
        "time_series_quality": ts_quality,
        "tabular_quality": tab_quality,
        "quality_confidence": conf["quality_confidence"],
        "confidence_level": level,
        "quality_components": conf["components"],
        "quality_explanation_summary": summary,
        "execution_mode": {"tiff_quality_mode": mode, "tiff_detailed_executed": run_tiff_detail},
    }
    if tiff_quality is not None:
        payload["tiff_quality"] = tiff_quality

    if artifacts and artifacts.get("enabled"):
        try:
            out_dir = Path(artifacts["output_dir"])
            artifacts["plots"].update(_plot_missingness_timeline_pngs(ts_quality, out_dir))
            tab_path = out_dir / "tabular_quality_overview.png"
            _plot_tabular_quality_overview_png(tab_quality, tab_path)
            if tab_path.exists():
                artifacts["plots"]["tabular_quality_overview_png"] = str(tab_path)
        except Exception as exc:
            artifacts.setdefault("warnings", []).append(f"quality_plot_generation_failed: {exc}")

    return payload


def _build_aux_vector_from_meta_row(row: Optional[pd.Series], aux_dim: int) -> np.ndarray:
    vec = np.zeros(aux_dim, dtype=np.float32)
    if row is None:
        return vec
    for i, col in enumerate(["lon", "lat", "year", "geoUncertaintyInM", "areaInM2"]):
        if i >= aux_dim:
            break
        if col in row.index:
            value = row.get(col, 0.0)
            vec[i] = float(value) if pd.notna(value) else 0.0
    return vec


def _build_malala_shap_batches(
    survey_id: int,
    env_dim: int,
    aux_dim: int,
    tensors: Dict[str, torch.Tensor],
    fixed_image: Optional[torch.Tensor] = None,
    n_batches: int = 40,
) -> List[Dict[str, torch.Tensor]]:
    env_table = _load_env_tables()
    meta_table = _load_meta_table()

    env_means = env_table.mean()
    env_stds = env_table.std().replace(0, 1)

    all_ids = [int(sid) for sid in env_table.index.tolist()]
    if not all_ids:
        return []

    rng = np.random.default_rng(int(survey_id) + 4242)
    sample_size = min(max(8, n_batches), len(all_ids))
    sampled = rng.choice(np.array(all_ids), size=sample_size, replace=False).tolist()
    if survey_id in all_ids and survey_id not in sampled:
        sampled[0] = survey_id

    fixed_landsat = tensors["x_landsat"].detach().cpu()
    fixed_climate = tensors["x_climate"].detach().cpu()

    batches: List[Dict[str, torch.Tensor]] = []
    for sid in sampled:
        if sid not in env_table.index:
            continue

        env_row = env_table.loc[sid]
        env_norm = ((env_row - env_means) / env_stds).fillna(0).values.astype(np.float32)
        if len(env_norm) > env_dim:
            env_norm = env_norm[:env_dim]
        elif len(env_norm) < env_dim:
            env_norm = np.pad(env_norm, (0, env_dim - len(env_norm)))

        meta_row = meta_table.loc[sid] if sid in meta_table.index else None
        aux_vec = _build_aux_vector_from_meta_row(meta_row, aux_dim)

        batches.append(
            {
                "env": torch.tensor(env_norm, dtype=torch.float32).unsqueeze(0),
                "aux": torch.tensor(aux_vec, dtype=torch.float32).unsqueeze(0),
                "landsat": fixed_landsat.clone(),
                "climate": fixed_climate.clone(),
            }
        )
        if fixed_image is not None:
            batches[-1]["image"] = fixed_image.clone()

    return batches


def _find_survey_tiff_path(survey_id: int) -> Tuple[Optional[Path], Optional[str]]:
    tiff_root_raw = (os.environ.get("TIFF_ROOT", "") or "").strip()
    tiff_root = Path(tiff_root_raw) if tiff_root_raw else (DATA_DIR / "SatelitePatches" / "PA-train")
    if not tiff_root.exists():
        return None, f"TIFF root not found: {tiff_root}"

    sid = str(int(survey_id))
    d1 = sid[-2:] if len(sid) >= 2 else sid.zfill(2)
    d2 = sid[-4:-2] if len(sid) >= 4 else "00"

    candidates = [
        tiff_root / d1 / d2 / f"{sid}.tiff",
        tiff_root / d1 / d2 / f"{sid}.tif",
        tiff_root / f"{sid}.tiff",
        tiff_root / f"{sid}.tif",
    ]
    tif_path = next((p for p in candidates if p.exists()), None)
    if tif_path is None:
        for p in tiff_root.rglob(f"{sid}.tif*"):
            tif_path = p
            break
    if tif_path is None:
        return None, f"No TIFF found for survey {survey_id}"
    return tif_path, None


def _load_survey_image_array(survey_id: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    tif_path, path_err = _find_survey_tiff_path(survey_id)
    if tif_path is None:
        return None, path_err or f"No TIFF found for survey {survey_id}"

    try:
        import rasterio  # type: ignore
    except Exception as exc:
        return None, f"rasterio unavailable: {exc}"

    try:
        with rasterio.open(tif_path) as ds:
            arr = ds.read()  # [C, H, W]
        if arr.ndim != 3:
            return None, f"Unexpected TIFF shape for survey {survey_id}: {arr.shape}"
        return arr.astype(np.float32), None
    except Exception as exc:
        return None, f"Failed to read TIFF for survey {survey_id}: {exc}"


def _load_survey_image_tensor(survey_id: int) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    arr, err = _load_survey_image_array(survey_id)
    if arr is None:
        return None, err

    try:
        if arr.shape[0] < 4:
            pad = np.zeros((4 - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > 4:
            arr = arr[:4]
        # Standardization per channel (robust)
        for c in range(arr.shape[0]):
            channel = arr[c]
            std = float(channel.std())
            if std < 1e-6:
                std = 1.0
            arr[c] = (channel - float(channel.mean())) / std
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0), None
    except Exception as exc:
        return None, f"Failed to convert TIFF for survey {survey_id}: {exc}"


def _resolve_tiff_quality_mode() -> str:
    mode = (os.environ.get(EXPLAIN_TIFF_QUALITY_MODE_ENV, "auto") or "").strip().lower()
    if mode in {"auto", "always", "off"}:
        return mode
    return "auto"


def _compute_tiff_quality(survey_id: int) -> Dict[str, Any]:
    arr, err = _load_survey_image_array(survey_id)
    if arr is None:
        return {"available": False, "error": err or "tiff_unavailable"}

    finite = np.isfinite(arr)
    invalid_ratio = float(1.0 - float(finite.sum()) / max(arr.size, 1))
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    mean_img = safe.mean(axis=0) if safe.ndim == 3 else safe
    p5 = float(np.percentile(mean_img, 5))
    p95 = float(np.percentile(mean_img, 95))
    denom = max(p95 - p5, 1e-6)
    norm_brightness = np.clip((mean_img - p5) / denom, 0.0, 1.0)

    grad_x = np.abs(np.diff(mean_img, axis=1))
    grad_y = np.abs(np.diff(mean_img, axis=0))
    gx = np.pad(grad_x, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(grad_y, ((0, 1), (0, 0)), mode="edge")
    local_contrast = (gx + gy) / 2.0
    norm_contrast = local_contrast / (denom + 1e-6)

    low_texture_ratio = float(np.mean(norm_contrast < 0.03))
    haze_cloud_proxy_ratio = float(np.mean((norm_brightness > 0.8) & (norm_contrast < 0.03)))

    return {
        "available": True,
        "invalid_pixel_ratio": round(_clamp01(invalid_ratio), 4),
        "low_texture_ratio": round(_clamp01(low_texture_ratio), 4),
        "haze_cloud_proxy_ratio": round(_clamp01(haze_cloud_proxy_ratio), 4),
        "method_note": (
            "Proxy image quality metrics (no official QA cloud mask): bright+low-contrast haze/cloud indicator."
        ),
    }


def _build_shap_feature_names(feature_names_env: List[str], env_dim: int, aux_dim: int) -> List[str]:
    env_names = feature_names_env[:env_dim]
    while len(env_names) < env_dim:
        env_names.append(f"env_feat_{len(env_names)}")

    aux_base = [
        "aux_lon",
        "aux_lat",
        "aux_year",
        "aux_geoUncertaintyInM",
        "aux_areaInM2",
    ]
    aux_names = aux_base[:aux_dim]
    while len(aux_names) < aux_dim:
        aux_names.append(f"aux_extra_{len(aux_names)}")

    return env_names + aux_names


def _run_malala_xai(
    *,
    survey_id: int,
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
    env_dim: int,
    aux_dim: int,
    feature_names_env: List[str],
    output_dir: Path,
    run_shap_override: Optional[bool] = None,
    run_attention_override: Optional[bool] = None,
    run_gradcam_override: Optional[bool] = None,
) -> Dict[str, Any]:
    enabled = _env_bool(EXPLAIN_ENABLE_MALALA_XAI_ENV, False)
    if not enabled:
        return {"enabled": False, "reason": f"{EXPLAIN_ENABLE_MALALA_XAI_ENV}=false"}

    module, err = _load_malala_xai_module()
    if err or module is None:
        return {"enabled": False, "reason": f"malala_backend_unavailable: {err}"}

    if not hasattr(module, "AdrianModelAdapter"):
        return {"enabled": False, "reason": "Malala explainability module missing AdrianModelAdapter."}

    run_shap = _env_bool(EXPLAIN_MALALA_RUN_SHAP_ENV, False) if run_shap_override is None else bool(run_shap_override)
    run_attention = _env_bool(EXPLAIN_MALALA_RUN_ATTENTION_ENV, False) if run_attention_override is None else bool(run_attention_override)
    run_gradcam = _env_bool(EXPLAIN_MALALA_RUN_GRADCAM_ENV, False) if run_gradcam_override is None else bool(run_gradcam_override)

    payload: Dict[str, Any] = {
        "enabled": True,
        "provider": "Malala/explainability.py",
        "plots": {},
        "warnings": [],
        "run_shap": run_shap,
        "run_attention": run_attention,
        "run_gradcam": run_gradcam,
    }

    try:
        adapter = module.AdrianModelAdapter()
    except Exception as exc:
        return {"enabled": False, "reason": f"adapter_init_failed: {exc}"}

    fixed_image: Optional[torch.Tensor] = None
    image_err: Optional[str] = None
    use_image_branch = bool(getattr(model, "use_image_branch", False))
    if use_image_branch:
        fixed_image, image_err = _load_survey_image_tensor(survey_id)
        if fixed_image is None and image_err:
            payload["warnings"].append(f"image_unavailable: {image_err}")

    batches = _build_malala_shap_batches(
        survey_id=survey_id,
        env_dim=env_dim,
        aux_dim=aux_dim,
        tensors=tensors,
        fixed_image=fixed_image if use_image_branch else None,
        n_batches=40,
    )
    if not batches:
        payload["warnings"].append("No batches available for Malala XAI backend.")
        return payload

    if run_shap:
        try:
            shap_dir = output_dir / "malala_shap"
            shap_dir.mkdir(parents=True, exist_ok=True)
            shap_feature_names = _build_shap_feature_names(feature_names_env, env_dim, aux_dim)
            shap_result, shap_logs = _run_quietly(
                module.shap_tabular,
                model=model,
                adapter=adapter,
                batches=batches,
                feature_names=shap_feature_names,
                n_background=min(30, len(batches)),
                n_explain=min(12, len(batches)),
                device=str(DEVICE),
                output_dir=str(shap_dir),
            )
            if shap_logs:
                payload["warnings"].append("SHAP backend logs captured (silenced for MCP stdio).")
            if shap_result is not None:
                imp = shap_dir / "shap_importance.png"
                dep = shap_dir / "shap_dependence.png"
                if imp.exists():
                    payload["plots"]["shap_importance_png"] = str(imp)
                if dep.exists():
                    payload["plots"]["shap_dependence_png"] = str(dep)
            else:
                payload["warnings"].append("SHAP skipped by backend (dependency or data constraints).")
        except Exception as exc:
            payload["warnings"].append(f"shap_failed: {exc}")

    if run_attention:
        try:
            attn_path = output_dir / "malala_landsat_attention.png"
            batch0 = batches[0]
            attn, attn_logs = _run_quietly(
                module.landsat_attention,
                model=model,
                adapter=adapter,
                batch=batch0,
                output_path=str(attn_path),
                device=str(DEVICE),
                year_labels=[f"{yr}" for yr in range(2000, 2021)],
            )
            if attn_logs:
                payload["warnings"].append("Attention backend logs captured (silenced for MCP stdio).")
            if attn is None:
                payload["warnings"].append("Malala attention returned None.")
            elif attn_path.exists():
                payload["plots"]["malala_landsat_attention_png"] = str(attn_path)
        except Exception as exc:
            payload["warnings"].append(f"attention_failed: {exc}")

    if run_gradcam:
        try:
            if not use_image_branch:
                payload["warnings"].append("GradCAM skipped: model has no image branch.")
            elif fixed_image is None:
                payload["warnings"].append("GradCAM skipped: image tensor unavailable.")
            else:
                grad_path = output_dir / "malala_gradcam.png"
                grad_attr, grad_logs = _run_quietly(
                    module.gradcam,
                    model=model,
                    adapter=adapter,
                    batch=batches[0],
                    species_idx=0,
                    device=str(DEVICE),
                    output_path=str(grad_path),
                )
                if grad_logs:
                    payload["warnings"].append("GradCAM backend logs captured (silenced for MCP stdio).")
                if grad_attr is None:
                    payload["warnings"].append("GradCAM returned None (dependency or layer constraints).")
                elif grad_path.exists():
                    payload["plots"]["malala_gradcam_png"] = str(grad_path)
        except Exception as exc:
            payload["warnings"].append(f"gradcam_failed: {exc}")

    return payload

# ──────────────────────────────────────────────────────────────
#  MAPPING speciesIdx → speciesId
# ──────────────────────────────────────────────────────────────

_META_CACHE: Optional[np.ndarray] = None
_TABULAR_REF_CACHE: Optional[Dict[str, Any]] = None

def _get_species_ids_from_meta() -> np.ndarray:
    """Lit les speciesId uniques triés depuis le CSV train."""
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE
    path = DATA_DIR / "GLC25_PA_metadata_train.csv"
    df = pd.read_csv(path, usecols=["speciesId"])
    ids = np.sort(df["speciesId"].dropna().astype(int).unique())
    _META_CACHE = ids
    return ids

# ══════════════════════════════════════════════════════════════
#  FONCTIONS MCP PUBLIQUES
# ══════════════════════════════════════════════════════════════

def list_available_checkpoints() -> Dict[str, Any]:
    """
    Liste les checkpoints disponibles et leurs scores.

    Returns {"ok": True, "checkpoints": [...], "checkpoints_dir": str}
    """
    ckpt_dir = _find_checkpoints_dir()
    if ckpt_dir is None:
        searched = [str(p) for p in _CKPT_CANDIDATES]
        return _error(
            "No checkpoints directory found",
            "Define CHECKPOINTS_DIR or verify checkpoint search paths.",
            searched_paths=searched,
        )
    ckpts = []
    for p in sorted(ckpt_dir.glob("*.pt")):
        try:
            meta = torch.load(str(p), map_location="cpu", weights_only=False)
            ckpts.append({
                "filename":  p.name,
                "fold":      meta.get("fold"),
                "score_f1":  round(float(meta.get("score", 0)), 5),
                "alpha":     round(float(meta.get("alpha", 1.4)), 2),
                "epoch":     meta.get("epoch"),
                "n_classes": meta.get("n_classes"),
            })
        except Exception:
            ckpts.append({"filename": p.name, "error": "unreadable"})

    return {
        "ok":              True,
        "checkpoints_dir": str(ckpt_dir),
        "default_checkpoint": _default_checkpoint_name(),
        "checkpoints":     ckpts,
    }


def explain_model_prediction(
    survey_id:        int,
    top_k:            int = 5,
    checkpoint_name:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Explication complète de la prédiction du modèle pour un site donné.

    Pour chaque espèce prédite, retourne :
      - Les variables environnementales les plus influentes (gradient × input)
      - La contribution de chaque famille de variables (BioClim, Sol, Élévation...)
      - Les saisons Landsat les plus importantes
      - Les mois Bioclim les plus importants

    Args:
        survey_id       : identifiant du site (ex: 212)
        top_k           : nombre d'espèces à expliquer (défaut: 5)
        checkpoint_name : nom du checkpoint. Si None, utilise
                          DEFAULT_EXPLAIN_CHECKPOINT (défaut: fold4_best.pt).

    Returns:
        {"ok": True, "survey_id": int, "predicted_species": [...],
         "species_explanations": [...], "branch_importance": {...},
         "model_info": {...}}
    """
    try:
        model, meta = _load_checkpoint(checkpoint_name)
    except Exception as e:
        return _error("Checkpoint load failed", str(e))

    try:
        tensors = _get_survey_tensors(
            survey_id, meta["env_dim"], meta["aux_dim"]
        )
    except Exception as e:
        return _error("Data preparation failed", str(e), survey_id=survey_id)

    # ── Inférence ─────────────────────────────────────────────
    with torch.no_grad():
        logits = model(
            tensors["x_env"],
            tensors["x_aux"],
            tensors["x_landsat"],
            tensors["x_climate"],
        )
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Top K espèces prédites
    top_indices = np.argsort(-probs)[:top_k].tolist()

    # Mapping index → speciesId
    try:
        species_ids = _get_species_ids_from_meta()
        predicted_species = [
            {
                "rank":       rank + 1,
                "species_idx":idx,
                "species_id": int(species_ids[idx]) if idx < len(species_ids) else idx,
                "probability":float(round(probs[idx], 4)),
                "confidence": "haute" if probs[idx] > 0.7 else
                              "moyenne" if probs[idx] > 0.4 else "faible",
            }
            for rank, idx in enumerate(top_indices)
        ]
    except Exception:
        predicted_species = [
            {"rank": r+1, "species_idx": idx, "probability": float(round(probs[idx], 4))}
            for r, idx in enumerate(top_indices)
        ]

    # ── Attribution gradient × input ──────────────────────────
    try:
        # Remettre requires_grad pour le calcul des gradients
        model.zero_grad()
        explanations = _gradient_x_input_attribution(
            model, tensors, top_indices
        )
        # Ajouter l'info espèce à chaque explication
        for i, exp in enumerate(explanations):
            if i < len(predicted_species):
                exp["species_id"]   = predicted_species[i].get("species_id")
                exp["probability"]  = predicted_species[i].get("probability")
    except Exception as e:
        explanations = [{"error": str(e)}]

    # ── Importance des branches ────────────────────────────────
    try:
        branch_imp = _branch_importance(model, tensors)
    except Exception as e:
        branch_imp = {"error": str(e)}

    # ── Poids d'attention ──────────────────────────────────────
    try:
        attn = _extract_attention_weights(model, tensors)
    except Exception as e:
        attn = {"error": str(e)}

    artifacts = _init_artifacts_payload("explain_model_prediction", int(survey_id))
    observation_quality: Dict[str, Any] = {}
    if artifacts.get("enabled"):
        try:
            out_dir = Path(artifacts["output_dir"])
            if isinstance(branch_imp, dict) and "error" not in branch_imp:
                path_branch = out_dir / "branch_importance.png"
                _plot_branch_importance_png(branch_imp, path_branch)
                artifacts["plots"]["branch_importance_png"] = str(path_branch)

            if isinstance(attn, dict) and "error" not in attn:
                artifacts["plots"].update(_plot_temporal_importance_pngs(attn, out_dir))

            if isinstance(explanations, list):
                species_plots = _plot_species_env_attribution_pngs(explanations, out_dir)
                if species_plots:
                    artifacts["plots"]["species_env_attributions"] = species_plots

            temporal_quality_source = attn if isinstance(attn, dict) and "error" not in attn else {}
            observation_quality = _compute_observation_quality(
                survey_id=int(survey_id),
                tensors=tensors,
                temporal_importance=temporal_quality_source,
                explanations=explanations if isinstance(explanations, list) else [],
                artifacts=artifacts,
            )

            malala_payload = _run_malala_xai(
                survey_id=int(survey_id),
                model=model,
                tensors=tensors,
                env_dim=int(meta["env_dim"]),
                aux_dim=int(meta["aux_dim"]),
                feature_names_env=list(tensors.get("feature_names", [])),
                output_dir=out_dir,
            )
            artifacts["malala_backend"] = malala_payload
        except Exception as e:
            artifacts.setdefault("warnings", []).append(f"plot_generation_failed: {e}")
    else:
        temporal_quality_source = attn if isinstance(attn, dict) and "error" not in attn else {}
        observation_quality = _compute_observation_quality(
            survey_id=int(survey_id),
            tensors=tensors,
            temporal_importance=temporal_quality_source,
            explanations=explanations if isinstance(explanations, list) else [],
            artifacts=None,
        )
    if not observation_quality:
        temporal_quality_source = attn if isinstance(attn, dict) and "error" not in attn else {}
        observation_quality = _compute_observation_quality(
            survey_id=int(survey_id),
            tensors=tensors,
            temporal_importance=temporal_quality_source,
            explanations=explanations if isinstance(explanations, list) else [],
            artifacts=None,
        )

    return {
        "ok":                  True,
        "survey_id":           int(survey_id),
        "predicted_species":   predicted_species,
        "species_explanations":explanations,
        "branch_importance":   branch_imp,
        "temporal_importance": attn,
        "observation_quality": observation_quality,
        "explainability_method_notes": {
            "tabular": "Gradient×Input (SHAP-like proxy, not exact SHAP).",
            "temporal": "Input-gradient saliency over Landsat/Bioclim timesteps (attention proxy).",
            "branch": "Relative L2 norm of branch embeddings.",
        },
        "artifacts": artifacts,
        "model_info": {
            "checkpoint":  meta["checkpoint_path"],
            "fold":        meta["fold"],
            "score_f1":    meta["score"],
            "n_classes":   meta["n_classes"],
        },
    }


def get_branch_importance(
    survey_id:       int,
    checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retourne la contribution de chaque branche du modèle pour un site.
    Branche = composant architectural (MLP env, MLP aux, Transformer Landsat, Transformer Bioclim).

    Returns {"ok": True, "branch_importance": {"Env. tabulaires": 34.2%, ...}}
    """
    try:
        model, meta = _load_checkpoint(checkpoint_name)
    except Exception as e:
        return _error("Checkpoint load failed", str(e))

    try:
        tensors = _get_survey_tensors(survey_id, meta["env_dim"], meta["aux_dim"])
    except Exception as e:
        return _error("Data preparation failed", str(e))

    try:
        branch_imp = _branch_importance(model, tensors)
        artifacts = _init_artifacts_payload("get_branch_importance", int(survey_id))
        if artifacts.get("enabled"):
            try:
                out_dir = Path(artifacts["output_dir"])
                path_branch = out_dir / "branch_importance.png"
                _plot_branch_importance_png(branch_imp, path_branch)
                artifacts["plots"]["branch_importance_png"] = str(path_branch)
            except Exception as e:
                artifacts.setdefault("warnings", []).append(f"plot_generation_failed: {e}")

        return {
            "ok":               True,
            "survey_id":        int(survey_id),
            "branch_importance":branch_imp,
            "interpretation":   (
                "Pourcentage de contribution de chaque branche du modèle "
                "à la décision finale (norme L2 de l'embedding)."
            ),
            "method_note": "Branch importance is derived from embedding norms (not Grad-CAM).",
            "artifacts": artifacts,
        }
    except Exception as e:
        return _error("Attribution failed", str(e))


def get_attention_weights(
    survey_id:       int,
    checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retourne l'importance de chaque saison Landsat et de chaque mois Bioclim
    pour la prédiction sur ce site.

    Returns {"ok": True, "landsat_seasonal_importance": [...], "bioclim_monthly_importance": [...]}
    """
    try:
        model, meta = _load_checkpoint(checkpoint_name)
    except Exception as e:
        return _error("Checkpoint load failed", str(e))

    try:
        tensors = _get_survey_tensors(survey_id, meta["env_dim"], meta["aux_dim"])
    except Exception as e:
        return _error("Data preparation failed", str(e))

    try:
        attn = _extract_attention_weights(model, tensors)
        artifacts = _init_artifacts_payload("get_attention_weights", int(survey_id))
        if artifacts.get("enabled"):
            try:
                out_dir = Path(artifacts["output_dir"])
                artifacts["plots"].update(_plot_temporal_importance_pngs(attn, out_dir))
                observation_quality = _compute_observation_quality(
                    survey_id=int(survey_id),
                    tensors=tensors,
                    temporal_importance=attn,
                    explanations=[],
                    artifacts=artifacts,
                )
                artifacts["malala_backend"] = _run_malala_xai(
                    survey_id=int(survey_id),
                    model=model,
                    tensors=tensors,
                    env_dim=int(meta["env_dim"]),
                    aux_dim=int(meta["aux_dim"]),
                    feature_names_env=list(tensors.get("feature_names", [])),
                    output_dir=out_dir,
                    run_shap_override=False,
                    run_attention_override=True,
                    run_gradcam_override=False,
                )
            except Exception as e:
                artifacts.setdefault("warnings", []).append(f"plot_generation_failed: {e}")
                observation_quality = _compute_observation_quality(
                    survey_id=int(survey_id),
                    tensors=tensors,
                    temporal_importance=attn,
                    explanations=[],
                    artifacts=None,
                )
        else:
            observation_quality = _compute_observation_quality(
                survey_id=int(survey_id),
                tensors=tensors,
                temporal_importance=attn,
                explanations=[],
                artifacts=None,
            )
        attn["ok"]        = True
        attn["survey_id"] = int(survey_id)
        attn["method_note"] = (
            "Temporal importance is computed from gradients over sequence inputs "
            "(adapted attention proxy, not exact Transformer attention map)."
        )
        attn["observation_quality"] = observation_quality
        attn["artifacts"] = artifacts
        return attn
    except Exception as e:
        return _error("Attention extraction failed", str(e))


# ──────────────────────────────────────────────────────────────
#  TEST RAPIDE
#  DATA_DIR=./data python tools_explain.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEST tools_explain.py")
    print("=" * 60)

    print("\n[1] list_available_checkpoints()")
    res = list_available_checkpoints()
    if res["ok"]:
        print(f"  dossier: {res['checkpoints_dir']}")
        for ck in res["checkpoints"]:
            print(f"  {ck.get('filename'):25s}  F1={ck.get('score_f1')}  fold={ck.get('fold')}")
    else:
        print(f"  ERREUR: {res['error']}")

    # Choisir un surveyId de test (premier du CSV si dispo)
    test_sid = 212
    meta_path = DATA_DIR / "GLC25_PA_metadata_train.csv"
    if meta_path.exists():
        df = pd.read_csv(meta_path, nrows=1)
        test_sid = int(df["surveyId"].iloc[0])
    print(f"\nSurveyId de test : {test_sid}")

    print("\n[2] get_branch_importance()")
    res = get_branch_importance(test_sid)
    if res["ok"]:
        for branch, pct in res["branch_importance"].items():
            bar = "█" * max(1, int(pct / 5))
            print(f"  {branch:40s} {bar} {pct:.1f}%")
    else:
        print(f"  ERREUR: {res['error']} - {res['hint']}")

    print("\n[3] get_attention_weights()")
    res = get_attention_weights(test_sid)
    if res["ok"]:
        print("  Saisons Landsat les plus importantes :")
        for s in res.get("landsat_seasonal_importance", [])[:3]:
            print(f"    {s['season']:15s}  {s['importance_pct']:.1f}%")
        print("  Mois Bioclim les plus importants :")
        for m in res.get("bioclim_monthly_importance", [])[:3]:
            print(f"    {m['month']:6s}  {m['importance_pct']:.1f}%")
    else:
        print(f"  ERREUR: {res['error']} - {res['hint']}")

    print("\n[4] explain_model_prediction() (top_k=3)")
    res = explain_model_prediction(test_sid, top_k=3)
    if res["ok"]:
        print(f"  Espèces prédites :")
        for sp in res["predicted_species"]:
            print(f"    #{sp['rank']} speciesId={sp.get('species_id')}  "
                  f"prob={sp['probability']:.3f}  confiance={sp.get('confidence')}")
        print(f"\n  Explication espèce #1 :")
        if res["species_explanations"]:
            exp = res["species_explanations"][0]
            print(f"    Familles de variables :")
            for fam, pct in exp.get("feature_families", {}).items():
                if pct > 0:
                    print(f"      {fam:25s} {pct:.1f}%")
            print(f"    Top features individuelles :")
            for f in exp.get("top_env_features", [])[:4]:
                print(f"      {f['feature']:15s} attr={f['attribution']:+.4f}  "
                      f"({f['direction']})")
        print(f"\n  Importance des branches :")
        for branch, pct in res.get("branch_importance", {}).items():
            bar = "█" * max(1, int(pct / 5))
            print(f"    {branch:40s} {bar} {pct:.1f}%")
    else:
        print(f"  ERREUR: {res['error']} - {res['hint']}")

    print("\n=== OK ===")
