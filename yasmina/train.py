"""
Adrian's model IMPROVED - 5-Fold Ensemble Training
====================================================

Ameliorations par rapport a l'original d'Adrian (0.200 public) :
  1. 5-fold ensemble avec moyenne des logits (au lieu d'un seul fold)
  2. CosineAnnealingLR scheduler (au lieu de LR fixe)
  3. Mixed precision AMP bfloat16 (2x plus rapide)
  4. Gradient clipping (max_norm=1.0, stabilise le training)
  5. num_workers=4 (au lieu de 0, 3x plus rapide en I/O)
  6. 50 epochs (au lieu de 30)
  7. Grille alpha plus fine pour calibration
  8. Warmup lineaire 3 epochs

Usage :
    # Entrainer les 5 folds (fait tout automatiquement)
    python3 train.py --all-folds --epochs 50 --batch-size 64 --gpu-id 1

    # Entrainer un seul fold
    python3 train.py --fold-index 0 --epochs 50 --batch-size 64 --gpu-id 1

    # Ensemble predict (apres avoir entraine les 5 folds)
    python3 train.py --ensemble-predict --gpu-id 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _apply_cuda_visible_devices_from_argv() -> None:
    gpu_id = None
    for index, token in enumerate(sys.argv):
        if token == "--gpu-id" and index + 1 < len(sys.argv):
            gpu_id = sys.argv[index + 1]
            break
        if token.startswith("--gpu-id="):
            gpu_id = token.split("=", 1)[1]
            break
    if gpu_id is not None and gpu_id.strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.strip()


_apply_cuda_visible_devices_from_argv()

# Limiter les threads CPU a 25% (8 threads sur 32 coeurs)
_CPU_BUDGET = max(1, int(os.cpu_count() * 0.25)) if os.cpu_count() else 4
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, str(_CPU_BUDGET))

import torch

torch.set_num_threads(_CPU_BUDGET)
torch.set_num_interop_threads(max(1, min(2, _CPU_BUDGET)))

from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MultimodalPlantModel


MONTHLY_PATTERN = re.compile(r"(.+)_\d{2}_\d{4}$")
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
              2.2, 2.5, 3.0, 3.5, 4.0, 5.0]
SERVER_BASE_DIR = Path(os.environ.get("GLC_DATA_DIR", "data"))
SERVER_LABELS_TRAIN = Path("~/GLC25_PA_metadata_train.csv").expanduser()
SERVER_LABELS_TEST = Path("~/GLC25_PA_metadata_test.csv").expanduser()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PreparedData:
    env_train: np.ndarray
    env_test: np.ndarray
    aux_train: np.ndarray
    aux_test: np.ndarray
    landsat_train_paths: List[Optional[str]]
    landsat_test_paths: List[Optional[str]]
    climate_train_paths: List[Optional[str]]
    climate_test_paths: List[Optional[str]]
    image_train_paths: List[Optional[str]]
    image_test_paths: List[Optional[str]]
    train_labels: List[np.ndarray]
    species_ids: np.ndarray
    train_survey_ids: np.ndarray
    test_survey_ids: np.ndarray
    metadata_train: pd.DataFrame
    has_test: bool


# ============================================================================
# ARGS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved Adrian model training")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true", help="Train ALL folds sequentially")
    parser.add_argument("--ensemble-predict", action="store_true", help="Ensemble predict from all fold models")
    parser.add_argument("--spatial-grid-size", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--branch-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--use-image-branch", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts_v4")
    parser.add_argument("--class-weight-mode", type=str, default="sqrt_inv",
                        choices=["none", "sqrt_inv", "log_inv"],
                        help="Class weighting: none, sqrt_inv (1/sqrt(freq)), log_inv (1/log(1+freq))")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for BCE (0=off, 0.05=recommended)")
    return parser.parse_args()


# ============================================================================
# UTILS
# ============================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demande mais indisponible.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATA RESOLUTION (identique Adrian)
# ============================================================================

def resolve_data_dir() -> Path:
    candidates = [Path("data"), Path(__file__).resolve().parents[1] / "data", SERVER_BASE_DIR]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing data directory. Checked: {[str(p) for p in candidates]}")


def resolve_environmental_values_dir() -> Path:
    return resolve_data_dir() / "EnvironmentalValues"


def resolve_pa_metadata_file(split: str) -> Path:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / f"GLC25_PA_metadata_{split}.csv",
        data_dir / f"GLC24_PA_metadata_{split}.csv",
        SERVER_LABELS_TRAIN if split == "train" else SERVER_LABELS_TEST,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing PA metadata file for split={split}")


def try_resolve_path(fn, *args):
    try:
        return fn(*args)
    except FileNotFoundError:
        return None


def resolve_patch_dir(split: str) -> Path:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / "PA" / f"PA-{split}",
        data_dir / "SatelitePatches" / f"PA-{split}",
        SERVER_BASE_DIR / "PA" / f"PA-{split}",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing patch directory for split={split}")


def resolve_landsat_dir(split: str) -> Optional[Path]:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / "SateliteTimeSeries-Landsat" / "cubes" / f"PA-{split}",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-landsat-time-series",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-landsat_time_series",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-landsat_time_serie",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-landsat-time-series",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-landsat_time_series",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_bioclim_cube_dir(split: str) -> Optional[Path]:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / "BioclimTimeSeries" / "cubes" / f"PA-{split}",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-bioclimatic_time_series",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-bioclimatic-time-series",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-bioclimatic_time_series",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-bioclimatic-time-series",
        SERVER_BASE_DIR / "GLC25-PA" / f"GLC25-PA-{split}-bioclimatic_time_series",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ============================================================================
# DATA PREPARATION (identique Adrian, legerement nettoye)
# ============================================================================

def sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if c != "surveyId"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_spatial_blocks(metadata: pd.DataFrame, grid_size: float) -> np.ndarray:
    lat_bins = np.floor(metadata["lat"].to_numpy(dtype=np.float32) / grid_size).astype(np.int32)
    lon_bins = np.floor(metadata["lon"].to_numpy(dtype=np.float32) / grid_size).astype(np.int32)
    return np.char.add(np.char.add(lat_bins.astype(str), "_"), lon_bins.astype(str))


def build_fold_masks(groups: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    return [np.isin(groups, fold_group) for fold_group in np.array_split(unique_groups, n_folds)]


def _load_metadata(is_train: bool):
    split = "train" if is_train else "test"
    metadata = pd.read_csv(resolve_pa_metadata_file(split))
    metadata["surveyId"] = metadata["surveyId"].astype(np.int32)
    metadata["lon"] = pd.to_numeric(metadata["lon"], errors="coerce")
    metadata["lat"] = pd.to_numeric(metadata["lat"], errors="coerce")
    metadata["year"] = pd.to_numeric(metadata["year"], errors="coerce")
    metadata["geoUncertaintyInM"] = pd.to_numeric(metadata["geoUncertaintyInM"], errors="coerce")
    metadata["areaInM2"] = pd.to_numeric(metadata["areaInM2"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if is_train:
        labels = metadata.groupby("surveyId")["speciesId"].apply(
            lambda v: np.sort(v.dropna().astype(np.int32).unique())
        ).reset_index(name="labels")
        return metadata.drop(columns=["speciesId"]).drop_duplicates("surveyId"), labels
    return metadata.drop_duplicates("surveyId"), None


def _load_env_table(relative_candidates: List[str]) -> pd.DataFrame:
    base_dir = resolve_environmental_values_dir()
    for rel_path in relative_candidates:
        candidate = base_dir / rel_path
        if candidate.exists():
            table = pd.read_csv(candidate)
            unnamed = [c for c in table.columns if str(c).startswith("Unnamed:")]
            if unnamed:
                table = table.drop(columns=unnamed)
            table["surveyId"] = table["surveyId"].astype(np.int32)
            return sanitize_numeric_frame(table.drop_duplicates("surveyId"))
    raise FileNotFoundError(f"Missing: {relative_candidates}")


def _standardize_pair(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_cols: Optional[List[str]] = None):
    train_df, test_df = train_df.copy(), test_df.copy()
    categorical_cols = categorical_cols or []
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for col in categorical_cols:
        combined[col] = combined[col].fillna("Unknown")
    if categorical_cols:
        combined = pd.get_dummies(combined, columns=categorical_cols, dtype=np.float32)
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    train_rows = len(train_df)
    train_arr = combined.iloc[:train_rows].to_numpy(dtype=np.float32)
    test_arr = combined.iloc[train_rows:].to_numpy(dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(train_arr, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_arr = np.where(np.isnan(train_arr), medians, train_arr)
    test_arr = np.where(np.isnan(test_arr), medians, test_arr)
    means = train_arr.mean(axis=0)
    stds = np.where(train_arr.std(axis=0) < 1e-6, 1.0, train_arr.std(axis=0))
    return ((train_arr - means) / stds).astype(np.float32), ((test_arr - means) / stds).astype(np.float32)


def _build_pt_index(directory: Path) -> Dict[int, str]:
    index: Dict[int, str] = {}
    for path in directory.rglob("*.pt"):
        for token in reversed(path.stem.split("_")):
            if token.isdigit():
                index[int(token)] = str(path)
                break
    return index


def _map_image_paths(metadata: pd.DataFrame, split: str) -> List[Optional[str]]:
    """Map survey IDs to satellite TIFF paths using directory structure.
    Structure: SatelitePatches/PA-{split}/{surveyId[-2:]}/{surveyId[-4:-2]}/{surveyId}.tiff
    """
    base_dir = Path("/data/challenge2026MIASHS/SatelitePatches") / f"PA-{split}"
    paths = []
    found = 0
    for sid in metadata["surveyId"].values:
        s = str(int(sid))
        d1 = s[-2:] if len(s) >= 2 else s.zfill(2)
        d2 = s[-4:-2] if len(s) >= 4 else s.zfill(4)[-4:-2]
        tiff_path = base_dir / d1 / d2 / f"{int(sid)}.tiff"
        if tiff_path.exists():
            paths.append(str(tiff_path))
            found += 1
        else:
            paths.append(None)
    print(f"  Image paths ({split}): found {found}/{len(metadata)} ({100*found/len(metadata):.1f}%)")
    return paths


def load_tensor_cube(path: str) -> Optional[torch.Tensor]:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError, OSError, ValueError, pickle.UnpicklingError):
        return None
    if isinstance(tensor, dict):
        tensor = next((v for v in tensor.values() if isinstance(v, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor):
        return None
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


def prepare_data(cache_path: Path) -> PreparedData:
    if cache_path.exists():
        print(f"  Loading cached data from {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    print("  Preparing data from scratch (will be cached)...")
    metadata_train, labels_train_df = _load_metadata(is_train=True)
    test_path = try_resolve_path(resolve_pa_metadata_file, "test")
    has_test = test_path is not None
    metadata_test, _ = _load_metadata(is_train=False) if has_test else (metadata_train.iloc[:0].copy(), None)

    def load_env(split):
        return (
            _load_env_table([f"ClimateAverage_1981-2010/GLC25-PA-{split}-bioclimatic.csv",
                            f"ClimateAverage_1981-2010/GLC24-PA-{split}-bioclimatic.csv"])
            .merge(_load_env_table([f"Elevation/GLC25-PA-{split}-elevation.csv",
                                    f"Elevation/GLC24-PA-{split}-elevation.csv"]), on="surveyId", how="left")
            .merge(_load_env_table([f"HumanFootprint/GLC25-PA-{split}-human_footprint.csv",
                                    f"HumanFootprint/GLC24-PA-{split}-human_footprint.csv"]), on="surveyId", how="left")
            .merge(_load_env_table([f"LandCover/GLC25-PA-{split}-landcover.csv",
                                    f"LandCover/GLC24-PA-{split}-landcover.csv"]), on="surveyId", how="left")
            .merge(_load_env_table([f"SoilGrids/GLC25-PA-{split}-soilgrids.csv",
                                    f"SoilGrids/GLC24-PA-{split}-soilgrids.csv"]), on="surveyId", how="left")
        )

    env_train_df = metadata_train[["surveyId"]].merge(load_env("train"), on="surveyId", how="left")
    env_test_df = metadata_test[["surveyId"]].merge(load_env("test"), on="surveyId", how="left") if has_test else env_train_df.iloc[:0].copy()
    env_train, env_test = _standardize_pair(env_train_df.drop(columns=["surveyId"]), env_test_df.drop(columns=["surveyId"]))
    aux_train, aux_test = _standardize_pair(
        metadata_train.drop(columns=["surveyId"]),
        metadata_test.drop(columns=["surveyId"]),
        categorical_cols=["region", "country"],
    )

    species_ids = np.array(sorted({int(sp) for labs in labels_train_df["labels"] for sp in labs}), dtype=np.int32)
    sp_to_idx = {sp: i for i, sp in enumerate(species_ids)}
    train_labels = [np.array([sp_to_idx[int(sp)] for sp in labs], dtype=np.int32) for labs in labels_train_df["labels"]]

    landsat_train_dir = resolve_landsat_dir("train")
    landsat_test_dir = resolve_landsat_dir("test") if has_test else None
    landsat_train_idx = _build_pt_index(landsat_train_dir) if landsat_train_dir else {}
    landsat_test_idx = _build_pt_index(landsat_test_dir) if landsat_test_dir else {}

    climate_train_dir = resolve_bioclim_cube_dir("train")
    climate_test_dir = resolve_bioclim_cube_dir("test") if has_test else None
    climate_train_idx = _build_pt_index(climate_train_dir) if climate_train_dir else {}
    climate_test_idx = _build_pt_index(climate_test_dir) if climate_test_dir else {}

    print(f"  Landsat train dir: {landsat_train_dir} ({len(landsat_train_idx)} cubes)")
    print(f"  Climate train dir: {climate_train_dir} ({len(climate_train_idx)} cubes)")

    prepared = PreparedData(
        env_train=env_train, env_test=env_test,
        aux_train=aux_train, aux_test=aux_test,
        landsat_train_paths=[landsat_train_idx.get(int(s)) for s in metadata_train["surveyId"]],
        landsat_test_paths=[landsat_test_idx.get(int(s)) for s in metadata_test["surveyId"]],
        climate_train_paths=[climate_train_idx.get(int(s)) for s in metadata_train["surveyId"]],
        climate_test_paths=[climate_test_idx.get(int(s)) for s in metadata_test["surveyId"]],
        image_train_paths=_map_image_paths(metadata_train, "train"),
        image_test_paths=_map_image_paths(metadata_test, "test") if has_test else [],
        train_labels=train_labels,
        species_ids=species_ids,
        train_survey_ids=metadata_train["surveyId"].to_numpy(dtype=np.int32),
        test_survey_ids=metadata_test["surveyId"].to_numpy(dtype=np.int32),
        metadata_train=metadata_train[["surveyId", "lon", "lat", "year", "region", "country"]].copy(),
        has_test=has_test,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(prepared, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached to {cache_path}")
    return prepared


# ============================================================================
# DATASET (identique Adrian)
# ============================================================================

class SurveyDataset(Dataset):
    def __init__(self, env_features, aux_features, landsat_paths, climate_paths,
                 image_paths, labels, n_classes, use_image_branch):
        self.env = env_features
        self.aux = aux_features
        self.landsat_paths = landsat_paths
        self.climate_paths = climate_paths
        self.image_paths = image_paths
        self.labels = labels
        self.n_classes = n_classes
        self.use_image_branch = use_image_branch

    def __len__(self):
        return len(self.env)

    @staticmethod
    def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        std = t.std()
        if torch.isnan(std) or std < 1e-6:
            std = torch.tensor(1.0, dtype=t.dtype)
        return (t - t.mean()) / std

    def _cube_to_series(self, cube, fallback_shape):
        if cube is None or cube.ndim != 3:
            return torch.zeros(fallback_shape, dtype=torch.float32)
        try:
            return self._normalize_tensor(cube.permute(2, 0, 1).reshape(fallback_shape[0], -1))
        except (RuntimeError, IndexError, ValueError):
            return torch.zeros(fallback_shape, dtype=torch.float32)

    def __getitem__(self, idx):
        lp = self.landsat_paths[idx]
        cp = self.climate_paths[idx]
        landsat = self._cube_to_series(None if lp is None else load_tensor_cube(lp), (21, 24))
        climate = self._cube_to_series(None if cp is None else load_tensor_cube(cp), (12, 76))
        image = None
        if self.use_image_branch:
            ip = self.image_paths[idx]
            if ip is None:
                image = torch.zeros((4, 64, 64), dtype=torch.float32)
            else:
                try:
                    import tifffile
                    arr = tifffile.imread(ip).astype(np.float32)  # (64, 64, 4) int16
                    arr = np.transpose(arr, (2, 0, 1))  # (4, 64, 64)
                    # Normalize Sentinel-2 reflectance: typical range 0-10000
                    # Scale to [0, 1] range then apply ImageNet-like normalization
                    arr = np.clip(arr, 0, 10000) / 10000.0
                    image = torch.from_numpy(arr)
                except Exception:
                    image = torch.zeros((4, 64, 64), dtype=torch.float32)
        return {
            "env": torch.from_numpy(self.env[idx]).float(),
            "aux": torch.from_numpy(self.aux[idx]).float(),
            "landsat": landsat,
            "climate": climate,
            "image": image,
            "labels": None if self.labels is None else self.labels[idx],
        }


def collate_fn(batch, n_classes):
    env = torch.stack([b["env"] for b in batch])
    aux = torch.stack([b["aux"] for b in batch])
    landsat = torch.stack([b["landsat"] for b in batch])
    climate = torch.stack([b["climate"] for b in batch])
    image = None if batch[0]["image"] is None else torch.stack([b["image"] for b in batch])
    targets = None
    if batch[0]["labels"] is not None:
        targets = torch.zeros((len(batch), n_classes), dtype=torch.float32)
        for i, b in enumerate(batch):
            targets[i, b["labels"]] = 1.0
    return env, aux, landsat, climate, image, targets


def move_to(batch, device):
    env, aux, landsat, climate, image, targets = batch
    nbc = device.type == "cuda"
    env = env.to(device, non_blocking=nbc)
    aux = aux.to(device, non_blocking=nbc)
    landsat = landsat.to(device, non_blocking=nbc)
    climate = climate.to(device, non_blocking=nbc)
    if image is not None:
        image = image.to(device, non_blocking=nbc)
    if targets is not None:
        targets = targets.to(device, non_blocking=nbc)
    return env, aux, landsat, climate, image, targets


# ============================================================================
# EVALUATION + CALIBRATION (identique Adrian, grille elargie)
# ============================================================================

def sample_f1_from_ranked_hits(cumhits, pred_counts, true_counts):
    safe = np.clip(pred_counts.astype(np.int32), 1, cumhits.shape[1])
    tp = cumhits[np.arange(cumhits.shape[0]), safe - 1]
    denom = safe + true_counts
    return float(np.where(denom > 0, (2.0 * tp) / denom, 0.0).mean())


def calibrate_alpha(prob_sums, cumhits, true_counts, min_k, max_k):
    safe_sums = np.nan_to_num(prob_sums, nan=0.0, posinf=0.0, neginf=0.0)
    best_alpha, best_score = ALPHA_GRID[0], -1.0
    for alpha in ALPHA_GRID:
        pred_counts = np.clip(np.rint(safe_sums * alpha).astype(np.int32), min_k, max_k)
        score = sample_f1_from_ranked_hits(cumhits, pred_counts, true_counts)
        if score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha, best_score


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device, min_k, max_k, amp_dtype):
    model.eval()
    losses, prob_sums_list, cumhits_list, true_counts_list = [], [], [], []
    for batch in loader:
        env, aux, landsat, climate, image, targets = move_to(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(env, aux, landsat, climate, image)
            loss = loss_fn(logits, targets)
        losses.append(loss.item())
        probs = torch.sigmoid(logits).float().cpu()
        tgt = targets.float().cpu()
        topk_idx = torch.topk(probs, k=max_k, dim=1).indices
        ranked_hits = torch.gather(tgt, 1, topk_idx)
        prob_sums_list.append(probs.sum(dim=1).numpy())
        cumhits_list.append(torch.cumsum(ranked_hits, dim=1).numpy())
        true_counts_list.append(tgt.sum(dim=1).numpy())
    alpha, score = calibrate_alpha(
        np.concatenate(prob_sums_list),
        np.concatenate(cumhits_list),
        np.concatenate(true_counts_list),
        min_k, max_k,
    )
    return float(np.mean(losses)), score, alpha


@torch.no_grad()
def predict_logits(model, loader, device, amp_dtype):
    """Retourne les logits bruts pour ensemble."""
    model.eval()
    all_logits = []
    for batch in loader:
        env, aux, landsat, climate, image, _ = move_to(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(env, aux, landsat, climate, image)
        all_logits.append(logits.float().cpu())
    return torch.cat(all_logits, dim=0)


# ============================================================================
# WARMUP + COSINE SCHEDULER
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)


# ============================================================================
# CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================================

def compute_class_weights(train_labels, n_classes, mode="sqrt_inv", device=None):
    """Compute per-class pos_weight for BCEWithLogitsLoss to handle class imbalance.

    Args:
        train_labels: list of arrays, each array = species indices present in a survey
        n_classes: total number of classes
        mode: 'sqrt_inv' = 1/sqrt(freq), 'log_inv' = 1/log(1+freq), 'none' = uniform
    """
    if mode == "none":
        return None

    # Count frequency of each class
    freq = np.zeros(n_classes, dtype=np.float64)
    for labels in train_labels:
        for idx in labels:
            freq[idx] += 1

    # Avoid division by zero
    freq = np.maximum(freq, 1.0)

    if mode == "sqrt_inv":
        weights = 1.0 / np.sqrt(freq)
    elif mode == "log_inv":
        weights = 1.0 / np.log1p(freq)
    else:
        return None

    # Normalize so mean weight = 1 (doesn't change relative weighting but keeps loss scale stable)
    weights = weights / weights.mean()

    # Clip weights conservatively to avoid destabilizing training
    weights = np.clip(weights, 0.5, 2.0)

    n_rare = (freq <= 10).sum()
    n_common = (freq > 1000).sum()
    print(f"  Class weights ({mode}): min={weights.min():.3f}, max={weights.max():.3f}, "
          f"mean={weights.mean():.3f}, rare(<=10)={n_rare}, common(>1000)={n_common}")

    w = torch.from_numpy(weights).float()
    if device is not None:
        w = w.to(device)
    return w


# ============================================================================
# TRAINING ONE FOLD
# ============================================================================

def train_fold(fold_idx, prepared, fold_masks, args, device):
    import time

    val_mask = fold_masks[fold_idx]
    train_mask = ~val_mask
    n_classes = len(prepared.species_ids)

    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx} / {args.n_folds - 1}")
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}")
    print(f"{'='*60}")

    train_ds = SurveyDataset(
        prepared.env_train[train_mask], prepared.aux_train[train_mask],
        [prepared.landsat_train_paths[i] for i in np.where(train_mask)[0]],
        [prepared.climate_train_paths[i] for i in np.where(train_mask)[0]],
        [prepared.image_train_paths[i] for i in np.where(train_mask)[0]],
        [prepared.train_labels[i] for i in np.where(train_mask)[0]],
        n_classes, args.use_image_branch,
    )
    val_ds = SurveyDataset(
        prepared.env_train[val_mask], prepared.aux_train[val_mask],
        [prepared.landsat_train_paths[i] for i in np.where(val_mask)[0]],
        [prepared.climate_train_paths[i] for i in np.where(val_mask)[0]],
        [prepared.image_train_paths[i] for i in np.where(val_mask)[0]],
        [prepared.train_labels[i] for i in np.where(val_mask)[0]],
        n_classes, args.use_image_branch,
    )

    cfn = lambda batch: collate_fn(batch, n_classes)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, collate_fn=cfn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin, collate_fn=cfn)

    model = MultimodalPlantModel(
        n_species=n_classes,
        env_dim=prepared.env_train.shape[1],
        aux_dim=prepared.aux_train.shape[1],
        landsat_features=24, climate_features=76,
        branch_dim=args.branch_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        use_image_branch=args.use_image_branch,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Class-weighted BCE for imbalanced data
    train_labels_fold = [prepared.train_labels[i] for i in np.where(train_mask)[0]]
    class_weights = compute_class_weights(train_labels_fold, n_classes, args.class_weight_mode, device)
    if class_weights is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    val_loss_fn = nn.BCEWithLogitsLoss()  # unweighted for comparable validation loss

    # Differential LR: pretrained image encoder gets lower LR for fine-tuning
    if args.use_image_branch and model.image_encoder is not None:
        image_params = list(model.image_encoder.parameters())
        image_param_ids = {id(p) for p in image_params}
        other_params = [p for p in model.parameters() if id(p) not in image_param_ids]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": args.lr},
            {"params": image_params, "lr": args.lr * 0.1},  # 10x lower for pretrained
        ], weight_decay=args.weight_decay)
        print(f"  Differential LR: main={args.lr}, image={args.lr*0.1}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    best_score, best_epoch, best_alpha, best_state = -1.0, -1, 1.0, None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        t0 = time.time()

        for batch in train_loader:
            env, aux, landsat, climate, image, targets = move_to(batch, device)

            # Label smoothing: push 0->eps and 1->(1-eps) to regularize
            if args.label_smoothing > 0:
                targets = targets * (1 - args.label_smoothing) + args.label_smoothing / 2

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(env, aux, landsat, climate, image)
                loss = loss_fn(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step(epoch)
        elapsed = time.time() - t0

        # Validation (use unweighted loss for fair comparison)
        val_loss, val_score, val_alpha = evaluate_model(
            model, val_loader, val_loss_fn, device, args.min_k, args.max_k, amp_dtype
        )

        is_best = val_score > best_score
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Fold{fold_idx} Ep{epoch+1:3d}/{args.epochs} | "
            f"TrLoss={np.mean(train_losses):.5f} VLoss={val_loss:.5f} "
            f"F1={val_score:.5f} alpha={val_alpha:.2f} lr={lr_now:.6f} "
            f"{elapsed:.0f}s {'*** BEST' if is_best else ''}"
        )

        if is_best:
            best_score, best_epoch, best_alpha = val_score, epoch + 1, val_alpha
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    save_path = args.output_dir / f"fold{fold_idx}_best.pt"
    torch.save({
        "state_dict": best_state,
        "fold": fold_idx,
        "epoch": best_epoch,
        "alpha": best_alpha,
        "score": best_score,
        "n_classes": n_classes,
        "env_dim": prepared.env_train.shape[1],
        "aux_dim": prepared.aux_train.shape[1],
        "branch_dim": args.branch_dim,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "dropout": args.dropout,
        "use_image_branch": args.use_image_branch,
    }, save_path)
    print(f"\n  Fold {fold_idx} DONE: best F1={best_score:.5f} at epoch {best_epoch}, alpha={best_alpha:.2f}")
    print(f"  Saved: {save_path}")
    return best_score, best_alpha


# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================

def ensemble_predict(args, prepared, device):
    """Charge les 5 fold models, moyenne les logits, genere submission."""
    n_classes = len(prepared.species_ids)
    cfn = lambda batch: collate_fn(batch, n_classes)
    pin = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # Test loader
    test_ds = SurveyDataset(
        prepared.env_test, prepared.aux_test,
        prepared.landsat_test_paths, prepared.climate_test_paths,
        prepared.image_test_paths, None, n_classes, args.use_image_branch,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin, collate_fn=cfn)

    # Charger et moyenner les logits de chaque fold
    all_logits_sum = None
    n_folds_loaded = 0
    alphas = []

    for fold_idx in range(args.n_folds):
        ckpt_path = args.output_dir / f"fold{fold_idx}_best.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping fold {fold_idx}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = MultimodalPlantModel(
            n_species=ckpt["n_classes"],
            env_dim=ckpt["env_dim"],
            aux_dim=ckpt["aux_dim"],
            landsat_features=24, climate_features=76,
            branch_dim=ckpt["branch_dim"],
            fusion_hidden_dim=ckpt["fusion_hidden_dim"],
            dropout=ckpt["dropout"],
            use_image_branch=ckpt["use_image_branch"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        alphas.append(ckpt["alpha"])
        print(f"  Loaded fold {fold_idx}: F1={ckpt['score']:.5f}, alpha={ckpt['alpha']:.2f}, epoch={ckpt['epoch']}")

        logits = predict_logits(model, test_loader, device, amp_dtype)
        if all_logits_sum is None:
            all_logits_sum = logits.double()
        else:
            all_logits_sum += logits.double()
        n_folds_loaded += 1
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if n_folds_loaded == 0:
        print("ERROR: No fold models found!")
        return

    avg_logits = all_logits_sum / n_folds_loaded
    probs = torch.sigmoid(avg_logits).numpy()
    avg_alpha = np.mean(alphas)

    print(f"\n  Ensemble: {n_folds_loaded} folds, avg_alpha={avg_alpha:.2f}")

    # Generer submissions pour different alphas
    for alpha in [avg_alpha, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5]:
        prob_sums = probs.sum(axis=1)
        counts = np.clip(np.rint(prob_sums * alpha).astype(int), args.min_k, args.max_k)
        topk_idx = np.argsort(-probs, axis=1)

        predictions = []
        for i in range(len(probs)):
            k = int(counts[i])
            species = np.sort(prepared.species_ids[topk_idx[i, :k]])
            predictions.append(" ".join(str(int(s)) for s in species))

        fname = f"submission_ens{n_folds_loaded}_alpha{alpha:.1f}.csv"
        out_path = args.output_dir / fname
        pd.DataFrame({
            "surveyId": prepared.test_survey_ids.astype(np.int32),
            "predictions": predictions,
        }).to_csv(out_path, index=False)
        print(f"  -> {out_path}")

    # Aussi generer avec Top-K fixe pour comparaison
    for K in [15, 20, 25]:
        topk_idx = np.argsort(-probs, axis=1)[:, :K]
        predictions = []
        for i in range(len(probs)):
            species = np.sort(prepared.species_ids[topk_idx[i]])
            predictions.append(" ".join(str(int(s)) for s in species))
        fname = f"submission_ens{n_folds_loaded}_topk{K}.csv"
        out_path = args.output_dir / fname
        pd.DataFrame({
            "surveyId": prepared.test_survey_ids.astype(np.int32),
            "predictions": predictions,
        }).to_csv(out_path, index=False)
        print(f"  -> {out_path}")

    print("\nDone!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Use different cache name when image branch is enabled (paths differ)
    cache_name = "prepared_with_images.pkl" if args.use_image_branch else "prepared_multimodal.pkl"
    print(f"\n[1] Preparing data...")
    prepared = prepare_data(cache_dir / cache_name)
    print(f"  Train: {len(prepared.train_survey_ids)}, Test: {len(prepared.test_survey_ids)}")
    print(f"  Species: {len(prepared.species_ids)}")
    print(f"  Env dim: {prepared.env_train.shape[1]}, Aux dim: {prepared.aux_train.shape[1]}")

    if args.ensemble_predict:
        print("\n[ENSEMBLE PREDICT]")
        ensemble_predict(args, prepared, device)
        return

    groups = build_spatial_blocks(prepared.metadata_train, args.spatial_grid_size)
    fold_masks = build_fold_masks(groups, args.n_folds, args.seed)

    if args.all_folds:
        print(f"\n[TRAINING ALL {args.n_folds} FOLDS]")
        results = {}
        for fold_idx in range(args.n_folds):
            score, alpha = train_fold(fold_idx, prepared, fold_masks, args, device)
            results[fold_idx] = (score, alpha)

        print(f"\n{'='*60}")
        print("RESUME ALL FOLDS")
        print(f"{'='*60}")
        for fi, (sc, al) in sorted(results.items()):
            print(f"  Fold {fi}: F1={sc:.5f}, alpha={al:.2f}")
        avg_score = np.mean([s for s, _ in results.values()])
        print(f"  Average F1: {avg_score:.5f}")

        # Auto-ensemble predict
        if prepared.has_test:
            print("\n[AUTO ENSEMBLE PREDICT]")
            ensemble_predict(args, prepared, device)
    else:
        train_fold(args.fold_index, prepared, fold_masks, args, device)


if __name__ == "__main__":
    main()
