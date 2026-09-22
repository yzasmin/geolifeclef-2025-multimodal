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

import torch
from pyproj import Transformer
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model import MultimodalPlantModel


MONTHLY_PATTERN = re.compile(r"(.+)_\d{2}_\d{4}$")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
ALPHA_GRID = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
SERVER_BASE_DIR = Path(os.environ.get("GLC_DATA_DIR", "data"))
SERVER_LABELS_TRAIN = Path("~/GLC25_PA_metadata_train.csv").expanduser()
SERVER_LABELS_TEST = Path("~/GLC25_PA_metadata_test.csv").expanduser()


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
    species_train_counts: np.ndarray
    landsat_mean: np.ndarray
    landsat_std: np.ndarray
    climate_mean: np.ndarray
    climate_std: np.ndarray


@dataclass
class TrainingArtifacts:
    best_alpha: float
    best_max_k: int
    best_score: float
    best_epoch: int
    thread_budget: int
    resource_fraction: float
    gpu_memory_fraction: float
    device: str
    use_image_branch: bool
    env_dim: int
    aux_dim: int
    n_species: int
    loss: str
    early_stopping_patience: int
    branch_dim: int
    fusion_hidden_dim: int
    dropout: float
    lr_scheduler: str
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    min_lr: float
    validation_mode: str
    sampler: str
    weighted_loss: bool
    freq_calibration: bool
    pos_weight_power: float
    pos_weight_max: float
    rare_sampler_power: float
    rare_sampler_max: float
    global_timeseries_norm: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal PA-only model.")
    parser.add_argument("--resource-fraction", type=float, default=0.25)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=str, default=None, help="Physical GPU id exposed through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--spatial-grid-size", type=float, default=1.0)
    parser.add_argument("--validation-mode", choices=("spatial", "group"), default="spatial")
    parser.add_argument("--holdout-group-column", choices=("region", "country"), default="region")
    parser.add_argument("--holdout-groups", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--branch-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--max-k-grid", type=str, default="30,40,50,70")
    parser.add_argument("--loss", choices=("asymmetric", "bce"), default="asymmetric")
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=30.0)
    parser.add_argument("--asym-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asym-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asym-clip", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--sampler", choices=("uniform", "rare"), default="rare")
    parser.add_argument("--rare-sampler-power", type=float, default=0.5)
    parser.add_argument("--rare-sampler-max", type=float, default=10.0)
    parser.add_argument("--freq-calibration", action="store_true")
    parser.add_argument("--freq-calibration-strength", type=float, default=0.12)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("plateau", "none"), default="plateau")
    parser.add_argument("--lr-scheduler-patience", type=int, default=2)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--use-image-branch", action="store_true")
    parser.add_argument("--safe-resource-mode", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "train_run")
    parser.add_argument("--ensemble-dirs", type=str, default="", help="Comma-separated fold output dirs containing best_model.pt and training_summary.json.")
    parser.add_argument("--ensemble-output", type=Path, default=None)
    parser.add_argument("--ensemble-alpha", type=float, default=None)
    parser.add_argument("--ensemble-max-k", type=int, default=None)
    return parser.parse_args()


def parse_max_k_grid(max_k_grid: str, fallback_max_k: int, min_k: int) -> List[int]:
    values: List[int] = []
    for token in max_k_grid.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value >= min_k:
            values.append(value)
    if fallback_max_k >= min_k:
        values.append(fallback_max_k)
    values = sorted(set(values))
    return values or [max(min_k, fallback_max_k)]


def compute_thread_budget(resource_fraction: float) -> int:
    cpu_count = os.cpu_count() or 1
    fraction = min(max(resource_fraction, 0.01), 1.0)
    return max(1, math.floor(cpu_count * fraction))


def apply_resource_limits(resource_fraction: float, cpu_only: bool) -> int:
    thread_budget = compute_thread_budget(resource_fraction)
    for env_var in THREAD_ENV_VARS:
        os.environ.setdefault(env_var, str(thread_budget))
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return thread_budget


def finalize_torch_limits(thread_budget: int, gpu_memory_fraction: Optional[float]) -> None:
    torch.set_num_threads(thread_budget)
    torch.set_num_interop_threads(max(1, min(2, thread_budget)))
    if torch.cuda.is_available() and gpu_memory_fraction is not None:
        fraction = min(max(gpu_memory_fraction, 0.01), 1.0)
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        except Exception:
            pass


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demande mais indisponible.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clamp_batch_size(batch_size: int, safe_resource_mode: bool) -> int:
    if not safe_resource_mode:
        return batch_size
    return max(8, min(batch_size, 64))


def resolve_data_dir() -> Path:
    candidates = [Path("data"), Path(__file__).resolve().parents[1] / "data", SERVER_BASE_DIR]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing data directory. Checked: {[str(path) for path in candidates]}")


def resolve_environmental_values_dir() -> Path:
    path = resolve_data_dir() / "EnvironmentalValues"
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing EnvironmentalValues directory: {path}")


def resolve_pa_metadata_file(split: str) -> Path:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / f"GLC25_PA_metadata_{split}.csv",
        data_dir / f"GLC24_PA_metadata_{split}.csv",
        SERVER_LABELS_TRAIN if split == "train" else SERVER_LABELS_TEST,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing PA metadata file for split={split}. Checked: {[str(path) for path in candidates]}")


def try_resolve_pa_metadata_file(split: str) -> Optional[Path]:
    try:
        return resolve_pa_metadata_file(split)
    except FileNotFoundError:
        return None


def resolve_patch_dir(split: str) -> Path:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / "PA" / f"PA-{split}",
        data_dir / "SatelitePatches" / f"PA-{split}",
        SERVER_BASE_DIR / "PA" / f"PA-{split}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing patch directory for split={split}. Checked: {[str(path) for path in candidates]}")


def resolve_landsat_dir(split: str) -> Path:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / "SateliteTimeSeries-Landsat" / "cubes" / f"PA-{split}",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-landsat-time-series",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-landsat_time_series",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-landsat_time_serie",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-landsat-time-series",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-landsat_time_series",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-landsat_time_serie",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing landsat directory for split={split}. Checked: {[str(path) for path in candidates]}")


def try_resolve_landsat_dir(split: str) -> Optional[Path]:
    try:
        return resolve_landsat_dir(split)
    except FileNotFoundError:
        return None


def resolve_bioclim_cube_dir(split: str) -> Path:
    data_dir = resolve_data_dir()
    candidates = [
        data_dir / "BioclimTimeSeries" / "cubes" / f"PA-{split}",
        data_dir / "GLC24-PA" / f"GLC24-PA-{split}-bioclimatic_time_series",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-bioclimatic_time_series",
        SERVER_BASE_DIR / "GLC24-PA" / f"GLC24-PA-{split}-bioclimatic-time-series",
        SERVER_BASE_DIR / "GLC25-PA" / f"GLC25-PA-{split}-bioclimatic_time_series",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing bioclim cube directory for split={split}")


def try_resolve_bioclim_cube_dir(split: str) -> Optional[Path]:
    try:
        return resolve_bioclim_cube_dir(split)
    except FileNotFoundError:
        return None


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


def build_group_holdout_mask(metadata: pd.DataFrame, column: str, raw_groups: str) -> np.ndarray:
    holdout_groups = {item.strip() for item in raw_groups.split(",") if item.strip()}
    if not holdout_groups:
        raise ValueError("validation-mode=group requires --holdout-groups.")
    return metadata[column].fillna("").astype(str).isin(holdout_groups).to_numpy()


def _load_metadata(is_train: bool) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    split = "train" if is_train else "test"
    metadata = pd.read_csv(resolve_pa_metadata_file(split))
    metadata["surveyId"] = metadata["surveyId"].astype(np.int32)
    metadata["lon"] = pd.to_numeric(metadata["lon"], errors="coerce")
    metadata["lat"] = pd.to_numeric(metadata["lat"], errors="coerce")
    metadata["year"] = pd.to_numeric(metadata["year"], errors="coerce")
    metadata["geoUncertaintyInM"] = pd.to_numeric(metadata["geoUncertaintyInM"], errors="coerce")
    metadata["areaInM2"] = pd.to_numeric(metadata["areaInM2"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if is_train:
        labels = metadata.groupby("surveyId")["speciesId"].apply(lambda values: np.sort(values.dropna().astype(np.int32).unique())).reset_index(name="labels")
        return metadata.drop(columns=["speciesId"]).drop_duplicates("surveyId"), labels
    return metadata.drop_duplicates("surveyId"), None


def _load_env_table(relative_candidates: List[str]) -> pd.DataFrame:
    base_dir = resolve_environmental_values_dir()
    table = None
    for rel_path in relative_candidates:
        candidate = base_dir / rel_path
        if candidate.exists():
            table = pd.read_csv(candidate)
            break
    if table is None:
        raise FileNotFoundError(f"Missing environmental table among: {relative_candidates}")
    unnamed_cols = [col for col in table.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        table = table.drop(columns=unnamed_cols)
    table["surveyId"] = table["surveyId"].astype(np.int32)
    return sanitize_numeric_frame(table.drop_duplicates("surveyId"))


def _standardize_pair(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    categorical_cols = categorical_cols or []
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for column in categorical_cols:
        combined[column] = combined[column].fillna("Unknown")
    if categorical_cols:
        combined = pd.get_dummies(combined, columns=categorical_cols, dtype=np.float32)
    for column in combined.columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    train_rows = len(train_df)
    train_array = combined.iloc[:train_rows].to_numpy(dtype=np.float32)
    test_array = combined.iloc[train_rows:].to_numpy(dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(train_array, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_array = np.where(np.isnan(train_array), medians, train_array)
    test_array = np.where(np.isnan(test_array), medians, test_array)
    means = train_array.mean(axis=0)
    stds = np.where(train_array.std(axis=0) < 1e-6, 1.0, train_array.std(axis=0))
    return ((train_array - means) / stds).astype(np.float32), ((test_array - means) / stds).astype(np.float32)


def _empty_like_rows(reference_df: pd.DataFrame) -> pd.DataFrame:
    return reference_df.iloc[:0].copy()


def compute_species_train_counts(train_labels: List[np.ndarray], n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    for labels in train_labels:
        if labels.size:
            counts[labels] += 1
    return counts


def compute_positive_class_weights(species_counts: np.ndarray, n_samples: int, power: float, max_weight: float) -> torch.Tensor:
    counts = np.maximum(species_counts.astype(np.float32), 1.0)
    neg_counts = np.maximum(float(n_samples) - counts, 1.0)
    weights = np.power(neg_counts / counts, power, dtype=np.float32)
    weights = np.clip(weights, 1.0, max_weight).astype(np.float32)
    return torch.from_numpy(weights)


def compute_survey_sampling_weights(train_labels: List[np.ndarray], species_counts: np.ndarray, power: float, max_weight: float) -> np.ndarray:
    counts = np.maximum(species_counts.astype(np.float32), 1.0)
    rarity = 1.0 / np.power(counts, power, dtype=np.float32)
    weights = np.ones(len(train_labels), dtype=np.float32)
    for index, labels in enumerate(train_labels):
        if labels.size:
            weights[index] = float(np.mean(rarity[labels]))
    weights = np.clip(weights / max(weights.mean(), 1e-8), 1e-3, max_weight)
    return weights.astype(np.float32)


def compute_species_frequency_adjustment(species_counts: np.ndarray, strength: float) -> np.ndarray:
    counts = np.maximum(species_counts.astype(np.float32), 1.0)
    log_counts = np.log1p(counts)
    centered = log_counts - float(np.median(log_counts))
    scale = np.max(np.abs(centered))
    if scale < 1e-8:
        return np.ones_like(counts, dtype=np.float32)
    normalized = centered / scale
    adjustment = 1.0 - strength * normalized
    return np.clip(adjustment, 0.75, 1.25).astype(np.float32)


def compute_global_sequence_stats(
    paths: List[Optional[str]],
    fallback_shape: Tuple[int, int],
    max_files: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_paths = [path for path in paths if path is not None]
    if not valid_paths:
        return np.zeros((fallback_shape[1],), dtype=np.float32), np.ones((fallback_shape[1],), dtype=np.float32)

    sample_paths = valid_paths if len(valid_paths) <= max_files else random.sample(valid_paths, max_files)
    sum_vec = np.zeros((fallback_shape[1],), dtype=np.float64)
    sumsq_vec = np.zeros((fallback_shape[1],), dtype=np.float64)
    count = 0

    for path in sample_paths:
        cube = load_tensor_cube(path)
        if cube is None or cube.ndim != 3:
            continue
        try:
            seq = cube.permute(2, 0, 1).reshape(fallback_shape[0], -1).numpy()
        except (RuntimeError, ValueError):
            continue
        if seq.shape[1] != fallback_shape[1]:
            continue
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        sum_vec += seq.sum(axis=0)
        sumsq_vec += np.square(seq).sum(axis=0)
        count += seq.shape[0]

    if count == 0:
        return np.zeros((fallback_shape[1],), dtype=np.float32), np.ones((fallback_shape[1],), dtype=np.float32)

    mean = sum_vec / count
    var = np.maximum(sumsq_vec / count - np.square(mean), 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _build_pt_index(directory: Path) -> Dict[int, str]:
    index: Dict[int, str] = {}
    for path in directory.rglob("*.pt"):
        for token in reversed(path.stem.split("_")):
            if token.isdigit():
                index[int(token)] = str(path)
                break
    return index


def _build_spatial_patch_lookup(directory: Path) -> Dict[Tuple[int, int], List[Tuple[str, float, float, float, float]]]:
    import rasterio

    buckets: Dict[Tuple[int, int], List[Tuple[str, float, float, float, float]]] = {}
    for path in directory.rglob("*.tiff"):
        with rasterio.open(path) as src:
            bounds = src.bounds
        record = (str(path), bounds.left, bounds.bottom, bounds.right, bounds.top)
        for grid_x in range(int(bounds.left // 10000), int(bounds.right // 10000) + 1):
            for grid_y in range(int(bounds.bottom // 10000), int(bounds.top // 10000) + 1):
                buckets.setdefault((grid_x, grid_y), []).append(record)
    return buckets


def _map_surveys_to_patch_paths(metadata: pd.DataFrame, patch_dir: Path) -> List[Optional[str]]:
    transformer = Transformer.from_crs("EPSG:4326", "IGNF:ETRS89LAEA", always_xy=True)
    buckets = _build_spatial_patch_lookup(patch_dir)
    paths: List[Optional[str]] = []
    for row in metadata.itertuples(index=False):
        x, y = transformer.transform(row.lon, row.lat)
        candidates = buckets.get((int(x // 10000), int(y // 10000)), [])
        found_path = None
        for patch_path, left, bottom, right, top in candidates:
            if left <= x <= right and bottom <= y <= top:
                found_path = patch_path
                break
        paths.append(found_path)
    return paths


def load_tensor_cube(path: str) -> Optional[torch.Tensor]:
    try:
        tensor = torch.load(path, map_location="cpu")
    except (EOFError, RuntimeError, OSError, ValueError, pickle.UnpicklingError):
        return None
    if isinstance(tensor, dict):
        tensor = next((value for value in tensor.values() if isinstance(value, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor):
        return None
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


def prepare_data(cache_path: Path) -> PreparedData:
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    metadata_train, labels_train_df = _load_metadata(is_train=True)
    test_metadata_path = try_resolve_pa_metadata_file("test")
    has_test = test_metadata_path is not None
    metadata_test, _ = _load_metadata(is_train=False) if has_test else (_empty_like_rows(metadata_train), None)

    def load_env(split: str) -> pd.DataFrame:
        return (
            _load_env_table([f"ClimateAverage_1981-2010/GLC25-PA-{split}-bioclimatic.csv", f"ClimateAverage_1981-2010/GLC24-PA-{split}-bioclimatic.csv"])
            .merge(_load_env_table([f"Elevation/GLC25-PA-{split}-elevation.csv", f"Elevation/GLC24-PA-{split}-elevation.csv"]), on="surveyId", how="left")
            .merge(_load_env_table([f"HumanFootprint/GLC25-PA-{split}-human_footprint.csv", f"HumanFootprint/GLC24-PA-{split}-human_footprint.csv"]), on="surveyId", how="left")
            .merge(_load_env_table([f"LandCover/GLC25-PA-{split}-landcover.csv", f"LandCover/GLC24-PA-{split}-landcover.csv"]), on="surveyId", how="left")
            .merge(_load_env_table([f"SoilGrids/GLC25-PA-{split}-soilgrids.csv", f"SoilGrids/GLC24-PA-{split}-soilgrids.csv"]), on="surveyId", how="left")
        )

    env_train_df = metadata_train[["surveyId"]].merge(load_env("train"), on="surveyId", how="left")
    env_test_df = metadata_test[["surveyId"]].merge(load_env("test"), on="surveyId", how="left") if has_test else _empty_like_rows(env_train_df)
    env_train, env_test = _standardize_pair(env_train_df.drop(columns=["surveyId"]), env_test_df.drop(columns=["surveyId"]))
    aux_train, aux_test = _standardize_pair(metadata_train.drop(columns=["surveyId"]), metadata_test.drop(columns=["surveyId"]), categorical_cols=["region", "country"])

    species_ids = np.array(sorted({int(species) for labels in labels_train_df["labels"] for species in labels}), dtype=np.int32)
    species_to_index = {species_id: index for index, species_id in enumerate(species_ids)}
    train_labels = [np.array([species_to_index[int(species)] for species in labels], dtype=np.int32) for labels in labels_train_df["labels"]]
    species_train_counts = compute_species_train_counts(train_labels, len(species_ids))

    landsat_train_dir = try_resolve_landsat_dir("train")
    landsat_test_dir = try_resolve_landsat_dir("test") if has_test else None
    if landsat_train_dir is None:
        print("Warning: Landsat train directory not found, using zero-filled Landsat tensors.")
    landsat_train_idx = _build_pt_index(landsat_train_dir) if landsat_train_dir is not None else {}
    landsat_test_idx = _build_pt_index(landsat_test_dir) if landsat_test_dir is not None else {}
    climate_train_dir = try_resolve_bioclim_cube_dir("train")
    climate_test_dir = try_resolve_bioclim_cube_dir("test") if has_test else None
    climate_train_idx = _build_pt_index(climate_train_dir) if climate_train_dir is not None else {}
    climate_test_idx = _build_pt_index(climate_test_dir) if climate_test_dir is not None else {}
    landsat_train_paths = [landsat_train_idx.get(int(sid)) for sid in metadata_train["surveyId"]]
    landsat_test_paths = [landsat_test_idx.get(int(sid)) for sid in metadata_test["surveyId"]]
    climate_train_paths = [climate_train_idx.get(int(sid)) for sid in metadata_train["surveyId"]]
    climate_test_paths = [climate_test_idx.get(int(sid)) for sid in metadata_test["surveyId"]]
    landsat_mean, landsat_std = compute_global_sequence_stats(landsat_train_paths, (21, 24))
    climate_mean, climate_std = compute_global_sequence_stats(climate_train_paths, (12, 76))

    prepared = PreparedData(
        env_train=env_train,
        env_test=env_test,
        aux_train=aux_train,
        aux_test=aux_test,
        landsat_train_paths=landsat_train_paths,
        landsat_test_paths=landsat_test_paths,
        climate_train_paths=climate_train_paths,
        climate_test_paths=climate_test_paths,
        image_train_paths=_map_surveys_to_patch_paths(metadata_train, resolve_patch_dir("train")),
        image_test_paths=_map_surveys_to_patch_paths(metadata_test, resolve_patch_dir("test")) if has_test else [],
        train_labels=train_labels,
        species_ids=species_ids,
        train_survey_ids=metadata_train["surveyId"].to_numpy(dtype=np.int32),
        test_survey_ids=metadata_test["surveyId"].to_numpy(dtype=np.int32),
        metadata_train=metadata_train[["surveyId", "lon", "lat", "year", "region", "country"]].copy(),
        has_test=has_test,
        species_train_counts=species_train_counts,
        landsat_mean=landsat_mean,
        landsat_std=landsat_std,
        climate_mean=climate_mean,
        climate_std=climate_std,
    )
    with cache_path.open("wb") as handle:
        pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return prepared


def sample_f1_from_ranked_hits(cumulative_hits: np.ndarray, predicted_counts: np.ndarray, true_counts: np.ndarray) -> float:
    safe_counts = np.clip(predicted_counts.astype(np.int32), 1, cumulative_hits.shape[1])
    tp = cumulative_hits[np.arange(cumulative_hits.shape[0]), safe_counts - 1]
    denom = safe_counts + true_counts
    return float(np.where(denom > 0, (2.0 * tp) / denom, 0.0).mean())


def calibrate_alpha_and_max_k(probability_sums: np.ndarray, cumulative_hits: np.ndarray, true_counts: np.ndarray, min_k: int, max_k_grid: List[int]) -> Tuple[float, int, float]:
    safe_sums = np.nan_to_num(probability_sums, nan=0.0, posinf=0.0, neginf=0.0)
    best_alpha, best_max_k, best_score = ALPHA_GRID[0], max_k_grid[0], -1.0
    for max_k in max_k_grid:
        for alpha in ALPHA_GRID:
            predicted_counts = np.clip(np.rint(safe_sums * alpha).astype(np.int32), min_k, max_k)
            score = sample_f1_from_ranked_hits(cumulative_hits, predicted_counts, true_counts)
            if score > best_score:
                best_alpha, best_max_k, best_score = alpha, max_k, score
    return best_alpha, best_max_k, best_score


class SurveyDataset(Dataset):
    def __init__(
        self,
        env_features,
        aux_features,
        landsat_paths,
        climate_paths,
        image_paths,
        labels,
        n_classes,
        use_image_branch,
        landsat_mean: np.ndarray,
        landsat_std: np.ndarray,
        climate_mean: np.ndarray,
        climate_std: np.ndarray,
    ):
        self.env_features = env_features
        self.aux_features = aux_features
        self.landsat_paths = landsat_paths
        self.climate_paths = climate_paths
        self.image_paths = image_paths
        self.labels = labels
        self.n_classes = n_classes
        self.use_image_branch = use_image_branch
        self.landsat_mean = torch.from_numpy(landsat_mean).float()
        self.landsat_std = torch.from_numpy(np.where(landsat_std < 1e-6, 1.0, landsat_std)).float()
        self.climate_mean = torch.from_numpy(climate_mean).float()
        self.climate_std = torch.from_numpy(np.where(climate_std < 1e-6, 1.0, climate_std)).float()

    def __len__(self) -> int:
        return len(self.env_features)

    @staticmethod
    def _normalize_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        mean = tensor.mean()
        std = tensor.std()
        if torch.isnan(std) or std < 1e-6:
            std = torch.tensor(1.0, dtype=tensor.dtype)
        return (tensor - mean) / std

    def _cube_to_series(
        self,
        cube: Optional[torch.Tensor],
        fallback_shape: Tuple[int, int],
        global_mean: torch.Tensor,
        global_std: torch.Tensor,
    ) -> torch.Tensor:
        if cube is None or cube.ndim != 3:
            return torch.zeros(fallback_shape, dtype=torch.float32)
        try:
            seq = cube.permute(2, 0, 1).reshape(fallback_shape[0], -1).float()
            seq = torch.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            return (seq - global_mean.unsqueeze(0)) / global_std.unsqueeze(0)
        except (RuntimeError, IndexError, ValueError):
            return torch.zeros(fallback_shape, dtype=torch.float32)

    def __getitem__(self, index: int):
        landsat_path = self.landsat_paths[index]
        climate_path = self.climate_paths[index]
        landsat_cube = None if landsat_path is None else load_tensor_cube(landsat_path)
        climate_cube = None if climate_path is None else load_tensor_cube(climate_path)
        landsat = self._cube_to_series(landsat_cube, (21, 24), self.landsat_mean, self.landsat_std)
        climate = self._cube_to_series(climate_cube, (12, 76), self.climate_mean, self.climate_std)
        image = None
        if self.use_image_branch:
            image_path = self.image_paths[index]
            if image_path is None:
                image = torch.zeros((4, 64, 64), dtype=torch.float32)
            else:
                import rasterio

                with rasterio.open(image_path) as src:
                    image = self._normalize_image_tensor(torch.from_numpy(src.read().astype(np.float32)))
        return {
            "env": torch.from_numpy(self.env_features[index]).float(),
            "aux": torch.from_numpy(self.aux_features[index]).float(),
            "landsat": landsat,
            "climate": climate,
            "image": image,
            "labels": None if self.labels is None else self.labels[index],
        }


def collate_fn(batch, n_classes: int):
    env = torch.stack([item["env"] for item in batch], dim=0)
    aux = torch.stack([item["aux"] for item in batch], dim=0)
    landsat = torch.stack([item["landsat"] for item in batch], dim=0)
    climate = torch.stack([item["climate"] for item in batch], dim=0)
    image = None if batch[0]["image"] is None else torch.stack([item["image"] for item in batch], dim=0)
    targets = None
    if batch[0]["labels"] is not None:
        targets = torch.zeros((len(batch), n_classes), dtype=torch.float32)
        for row_idx, item in enumerate(batch):
            targets[row_idx, item["labels"]] = 1.0
    return env, aux, landsat, climate, image, targets


def move_batch_to_device(batch, device):
    env, aux, landsat, climate, image, targets = batch
    env = env.to(device, non_blocking=device.type == "cuda")
    aux = aux.to(device, non_blocking=device.type == "cuda")
    landsat = landsat.to(device, non_blocking=device.type == "cuda")
    climate = climate.to(device, non_blocking=device.type == "cuda")
    if image is not None:
        image = image.to(device, non_blocking=device.type == "cuda")
    if targets is not None:
        targets = targets.to(device, non_blocking=device.type == "cuda")
    return env, aux, landsat, climate, image, targets


def evaluate_model(model, loader, loss_fn, device, min_k: int, max_k_grid: List[int], species_adjustment: Optional[np.ndarray] = None):
    model.eval()
    losses, probability_sums, cumulative_hits, true_counts = [], [], [], []
    top_k_limit = max(max_k_grid)
    adjustment_tensor = None if species_adjustment is None else torch.from_numpy(species_adjustment).float()
    with torch.no_grad():
        for batch in loader:
            env, aux, landsat, climate, image, targets = move_batch_to_device(batch, device)
            logits = model(env, aux, landsat, climate, image)
            loss = loss_fn(logits, targets)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu()
            if adjustment_tensor is not None:
                probs = torch.clamp(probs * adjustment_tensor.unsqueeze(0), 0.0, 1.0)
            targets_cpu = targets.cpu()
            topk_idx = torch.topk(probs, k=top_k_limit, dim=1).indices
            ranked_hits = torch.gather(targets_cpu, 1, topk_idx)
            probability_sums.append(probs.sum(dim=1).numpy())
            cumulative_hits.append(torch.cumsum(ranked_hits, dim=1).numpy())
            true_counts.append(targets_cpu.sum(dim=1).numpy())
    alpha, max_k, score = calibrate_alpha_and_max_k(np.concatenate(probability_sums), np.concatenate(cumulative_hits, axis=0), np.concatenate(true_counts), min_k, max_k_grid)
    return float(np.mean(losses)), score, alpha, max_k


def predict_test(model, loader, device, alpha: float, min_k: int, max_k: int, species_ids: np.ndarray, species_adjustment: Optional[np.ndarray] = None) -> list[str]:
    model.eval()
    predictions = []
    adjustment_tensor = None if species_adjustment is None else torch.from_numpy(species_adjustment).float()
    with torch.no_grad():
        for batch in loader:
            env, aux, landsat, climate, image, _ = move_batch_to_device(batch, device)
            probs = torch.sigmoid(model(env, aux, landsat, climate, image)).cpu()
            if adjustment_tensor is not None:
                probs = torch.clamp(probs * adjustment_tensor.unsqueeze(0), 0.0, 1.0)
            counts = torch.clamp(torch.round(probs.sum(dim=1) * alpha).to(torch.int64), min=min_k, max=max_k)
            topk_idx = torch.topk(probs, k=max_k, dim=1).indices.numpy()
            for row_idx in range(topk_idx.shape[0]):
                species = np.sort(species_ids[topk_idx[row_idx, : int(counts[row_idx].item())]])
                predictions.append(" ".join(str(int(species_id)) for species_id in species))
    return predictions


def _parse_ensemble_dirs(raw_dirs: str) -> List[Path]:
    return [Path(item.strip()) for item in raw_dirs.split(",") if item.strip()]


def _read_training_summary(output_dir: Path) -> Dict[str, object]:
    summary_path = output_dir / "training_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_model_from_summary(summary: Dict[str, object], prepared: PreparedData, n_classes: int, args: argparse.Namespace) -> MultimodalPlantModel:
    return MultimodalPlantModel(
        n_species=n_classes,
        env_dim=int(summary.get("env_dim", prepared.env_train.shape[1])),
        aux_dim=int(summary.get("aux_dim", prepared.aux_train.shape[1])),
        landsat_features=24,
        climate_features=76,
        branch_dim=int(summary.get("branch_dim", args.branch_dim)),
        fusion_hidden_dim=int(summary.get("fusion_hidden_dim", args.fusion_hidden_dim)),
        dropout=float(summary.get("dropout", args.dropout)),
        use_image_branch=bool(summary.get("use_image_branch", args.use_image_branch)),
    )


def predict_test_ensemble(
    model_dirs: List[Path],
    loader,
    device,
    species_ids: np.ndarray,
    prepared: PreparedData,
    args: argparse.Namespace,
) -> Tuple[List[str], float, int]:
    models = []
    alphas = []
    max_ks = []
    n_classes = len(species_ids)
    for model_dir in model_dirs:
        checkpoint_path = model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing ensemble checkpoint: {checkpoint_path}")
        summary = _read_training_summary(model_dir)
        model = _build_model_from_summary(summary, prepared, n_classes, args).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        if "best_alpha" in summary:
            alphas.append(float(summary["best_alpha"]))
        if "best_max_k" in summary:
            max_ks.append(int(summary["best_max_k"]))

    alpha = args.ensemble_alpha if args.ensemble_alpha is not None else float(np.mean(alphas) if alphas else 1.0)
    max_k = args.ensemble_max_k if args.ensemble_max_k is not None else int(np.median(max_ks) if max_ks else args.max_k)
    max_k = max(args.min_k, min(max_k, len(species_ids)))
    species_adjustment = compute_species_frequency_adjustment(prepared.species_train_counts, args.freq_calibration_strength) if args.freq_calibration else None
    adjustment_tensor = None if species_adjustment is None else torch.from_numpy(species_adjustment).float()

    predictions = []
    with torch.no_grad():
        for batch in loader:
            env, aux, landsat, climate, image, _ = move_batch_to_device(batch, device)
            probs_sum = None
            for model in models:
                probs = torch.sigmoid(model(env, aux, landsat, climate, image)).cpu()
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs_avg = probs_sum / len(models)
            if adjustment_tensor is not None:
                probs_avg = torch.clamp(probs_avg * adjustment_tensor.unsqueeze(0), 0.0, 1.0)
            counts = torch.clamp(torch.round(probs_avg.sum(dim=1) * alpha).to(torch.int64), min=args.min_k, max=max_k)
            topk_idx = torch.topk(probs_avg, k=max_k, dim=1).indices.numpy()
            for row_idx in range(topk_idx.shape[0]):
                species = np.sort(species_ids[topk_idx[row_idx, : int(counts[row_idx].item())]])
                predictions.append(" ".join(str(int(species_id)) for species_id in species))
    return predictions, alpha, max_k


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, label_smoothing: float = 0.03, pos_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smoothing = label_smoothing
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.float())
        else:
            self.pos_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + (1.0 - targets) * (self.label_smoothing / 2.0)

        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos.clamp(min=1e-8))
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=1e-8))
        if self.pos_weights is not None:
            loss_pos = loss_pos * self.pos_weights.unsqueeze(0)

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow((1.0 - pt).clamp(min=0.0), gamma)
            loss_pos = loss_pos * focal_weight
            loss_neg = loss_neg * focal_weight

        return -(loss_pos + loss_neg).mean()


def build_loss(args: argparse.Namespace, pos_weights: Optional[torch.Tensor] = None) -> nn.Module:
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    return AsymmetricLoss(
        gamma_pos=args.asym_gamma_pos,
        gamma_neg=args.asym_gamma_neg,
        clip=args.asym_clip,
        label_smoothing=args.label_smoothing,
        pos_weights=pos_weights,
    )


def build_test_loader(prepared: PreparedData, n_classes: int, args: argparse.Namespace, use_pin_memory: bool):
    test_dataset = SurveyDataset(
        prepared.env_test,
        prepared.aux_test,
        prepared.landsat_test_paths,
        prepared.climate_test_paths,
        prepared.image_test_paths,
        None,
        n_classes,
        args.use_image_branch,
        prepared.landsat_mean,
        prepared.landsat_std,
        prepared.climate_mean,
        prepared.climate_std,
    )
    return DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory, collate_fn=lambda batch: collate_fn(batch, n_classes))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_k_grid = parse_max_k_grid(args.max_k_grid, args.max_k, args.min_k)

    args.batch_size = clamp_batch_size(args.batch_size, args.safe_resource_mode)
    thread_budget = apply_resource_limits(args.resource_fraction, cpu_only=(args.device == "cpu"))
    finalize_torch_limits(thread_budget, args.gpu_memory_fraction if args.device != "cpu" else None)
    device = resolve_device(args.device)

    prepared = prepare_data(cache_dir / "prepared_multimodal.pkl")
    n_classes = len(prepared.species_ids)

    use_pin_memory = device.type == "cuda" and not args.safe_resource_mode
    if args.ensemble_dirs:
        if not prepared.has_test:
            raise RuntimeError("Ensemble prediction requires test metadata/data.")
        model_dirs = _parse_ensemble_dirs(args.ensemble_dirs)
        summaries = [_read_training_summary(model_dir) for model_dir in model_dirs]
        args.use_image_branch = any(bool(summary.get("use_image_branch", args.use_image_branch)) for summary in summaries)
        test_loader = build_test_loader(prepared, n_classes, args, use_pin_memory)
        submission, ensemble_alpha, ensemble_max_k = predict_test_ensemble(model_dirs, test_loader, device, prepared.species_ids, prepared, args)
        output_path = args.ensemble_output or (args.output_dir / "ensemble_submission.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"surveyId": prepared.test_survey_ids.astype(np.int32), "predictions": submission}).to_csv(output_path, index=False)
        print(f"Saved ensemble submission to {output_path}")
        print(f"Ensemble models: {len(model_dirs)} | alpha={ensemble_alpha:.2f} | max_k={ensemble_max_k}")
        return

    if args.validation_mode == "group":
        val_mask = build_group_holdout_mask(prepared.metadata_train, args.holdout_group_column, args.holdout_groups)
    else:
        groups = build_spatial_blocks(prepared.metadata_train, args.spatial_grid_size)
        fold_masks = build_fold_masks(groups, args.n_folds, args.seed)
        val_mask = fold_masks[args.fold_index]
    train_mask = ~val_mask
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise RuntimeError("Invalid train/validation split: one side is empty.")

    train_labels_subset = [prepared.train_labels[i] for i in np.where(train_mask)[0]]
    train_species_counts = compute_species_train_counts(train_labels_subset, n_classes)
    species_adjustment = compute_species_frequency_adjustment(train_species_counts, args.freq_calibration_strength) if args.freq_calibration else None

    train_dataset = SurveyDataset(
        prepared.env_train[train_mask],
        prepared.aux_train[train_mask],
        [prepared.landsat_train_paths[i] for i in np.where(train_mask)[0]],
        [prepared.climate_train_paths[i] for i in np.where(train_mask)[0]],
        [prepared.image_train_paths[i] for i in np.where(train_mask)[0]],
        train_labels_subset,
        n_classes,
        args.use_image_branch,
        prepared.landsat_mean,
        prepared.landsat_std,
        prepared.climate_mean,
        prepared.climate_std,
    )
    val_dataset = SurveyDataset(
        prepared.env_train[val_mask],
        prepared.aux_train[val_mask],
        [prepared.landsat_train_paths[i] for i in np.where(val_mask)[0]],
        [prepared.climate_train_paths[i] for i in np.where(val_mask)[0]],
        [prepared.image_train_paths[i] for i in np.where(val_mask)[0]],
        [prepared.train_labels[i] for i in np.where(val_mask)[0]],
        n_classes,
        args.use_image_branch,
        prepared.landsat_mean,
        prepared.landsat_std,
        prepared.climate_mean,
        prepared.climate_std,
    )
    train_sampler = None
    train_shuffle = True
    if args.sampler == "rare":
        survey_weights = compute_survey_sampling_weights(train_labels_subset, train_species_counts, args.rare_sampler_power, args.rare_sampler_max)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(survey_weights, dtype=torch.double),
            num_samples=len(survey_weights),
            replacement=True,
        )
        train_shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=0, pin_memory=use_pin_memory, collate_fn=lambda batch: collate_fn(batch, n_classes))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory, collate_fn=lambda batch: collate_fn(batch, n_classes))
    test_loader = None
    if prepared.has_test:
        test_loader = build_test_loader(prepared, n_classes, args, use_pin_memory)

    model = MultimodalPlantModel(
        n_species=n_classes,
        env_dim=prepared.env_train.shape[1],
        aux_dim=prepared.aux_train.shape[1],
        landsat_features=24,
        climate_features=76,
        branch_dim=args.branch_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        use_image_branch=args.use_image_branch,
    ).to(device)

    pos_weights = compute_positive_class_weights(train_species_counts, len(train_labels_subset), args.pos_weight_power, args.pos_weight_max) if args.weighted_loss else None
    loss_fn = build_loss(args, pos_weights=pos_weights.to(device) if pos_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.min_lr,
        )

    best_score, best_epoch, best_alpha, best_max_k, best_state = -1.0, -1, 1.0, args.max_k, None
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            env, aux, landsat, climate, image, targets = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(env, aux, landsat, climate, image), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_score, val_alpha, val_max_k = evaluate_model(model, val_loader, loss_fn, device, args.min_k, max_k_grid, species_adjustment=species_adjustment)
        if scheduler is not None:
            scheduler.step(val_score)
        is_best = val_score > best_score + args.early_stopping_min_delta
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1:02d}/{args.epochs:02d}] "
            f"train_loss={np.mean(train_losses):.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_f1={val_score:.5f} "
            f"alpha={val_alpha:.2f} "
            f"max_k={val_max_k} "
            f"lr={current_lr:.2e} "
            f"{'BEST' if is_best else ''}".rstrip()
        )
        if is_best:
            best_score, best_epoch, best_alpha, best_max_k = val_score, epoch + 1, val_alpha, val_max_k
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping after {epoch + 1} epochs "
                    f"(best_epoch={best_epoch}, best_val_f1={best_score:.5f})."
                )
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, args.output_dir / "best_model.pt")

    if prepared.has_test and test_loader is not None:
        submission = predict_test(model, test_loader, device, best_alpha, args.min_k, best_max_k, prepared.species_ids, species_adjustment=species_adjustment)
        pd.DataFrame({"surveyId": prepared.test_survey_ids.astype(np.int32), "predictions": submission}).to_csv(args.output_dir / "submission.csv", index=False)

    artifacts = TrainingArtifacts(
        best_alpha=best_alpha,
        best_max_k=best_max_k,
        best_score=best_score,
        best_epoch=best_epoch,
        thread_budget=thread_budget,
        resource_fraction=args.resource_fraction,
        gpu_memory_fraction=args.gpu_memory_fraction,
        device=str(device),
        use_image_branch=args.use_image_branch,
        env_dim=prepared.env_train.shape[1],
        aux_dim=prepared.aux_train.shape[1],
        n_species=n_classes,
        loss=args.loss,
        early_stopping_patience=args.early_stopping_patience,
        branch_dim=args.branch_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        min_lr=args.min_lr,
        validation_mode=args.validation_mode,
        sampler=args.sampler,
        weighted_loss=args.weighted_loss,
        freq_calibration=args.freq_calibration,
        pos_weight_power=args.pos_weight_power,
        pos_weight_max=args.pos_weight_max,
        rare_sampler_power=args.rare_sampler_power,
        rare_sampler_max=args.rare_sampler_max,
        global_timeseries_norm=True,
    )
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifacts), handle, indent=2)

    print(f"Best epoch: {best_epoch} | Best val_f1: {best_score:.5f} | Best alpha: {best_alpha:.2f} | Best max_k: {best_max_k}")
    print(f"Saved best model to {args.output_dir / 'best_model.pt'}")
    if prepared.has_test:
        print(f"Saved submission to {args.output_dir / 'submission.csv'}")
    else:
        print("Test metadata absent: skipped submission generation.")
    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
