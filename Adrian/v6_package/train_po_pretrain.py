"""
PO Pre-training + PA Fine-tuning (v7)
======================================
Improvements over v6:
  - FULL 57-dim aux features for PO (assign region/country via nearest PA survey)
  - 15 epochs PO pre-training (was 5 → underfitting)
  - Label smoothing 0.1 on PO cross-entropy
  - Complete weight transfer (no partial mismatch)

Phase 1: Pre-train environment + aux encoders on 3.5M PO observations
Phase 2: Fine-tune full model on PA data (transfer pre-trained weights)
"""

# ---------------------------------------------------------------------------
# Variantes testees puis retirees (gardees ici en memoire uniquement)
# ---------------------------------------------------------------------------
# - Alignement plus strict des chemins avec le repo local / ./data
# - Variantes de cache et de sorties sous Adrian/v6_package/artifacts
# - Reutilisation de backbones image alternatifs via model.py
# Cette version doit rester strictement alignee avec la reference fournie
# dans Downloads.

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
import pickle
import random
import warnings
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from functools import partial

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path(os.environ.get("GLC_DATA_DIR", "data"))
ENV_DIR = DATA_DIR / "EnvironmentalValues"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PO_PRETRAIN_EPOCHS = 15
PO_PRETRAIN_LR = 1e-3
PO_BATCH_SIZE = 512
PA_EPOCHS = 20
PA_LR = 5e-4
PA_BATCH_SIZE = 64

SEED = 42
N_FOLDS = 5
BRANCH_DIM = 128
FUSION_HIDDEN_DIM = 512

# ============================================================================
# SEED
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# DATA LOADING UTILS
# ============================================================================

def sanitize_numeric_frame(df):
    for col in [c for c in df.columns if c != "surveyId"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_env_table(prefix, split):
    """Load environment table for a given prefix (PA/PO) and split."""
    mapping = {
        "bioclimatic": f"ClimateAverage_1981-2010/GLC25-{prefix}-{split}-bioclimatic.csv",
        "elevation": f"Elevation/GLC25-{prefix}-{split}-elevation.csv",
        "human_footprint": f"HumanFootprint/GLC25-{prefix}-{split}-human_footprint.csv",
        "landcover": f"LandCover/GLC25-{prefix}-{split}-landcover.csv",
        "soilgrids": f"SoilGrids/GLC25-{prefix}-{split}-soilgrids.csv",
    }

    result = None
    for name, rel_path in mapping.items():
        path = ENV_DIR / rel_path
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        table = pd.read_csv(path)
        unnamed = [c for c in table.columns if str(c).startswith("Unnamed:")]
        if unnamed:
            table = table.drop(columns=unnamed)
        table["surveyId"] = table["surveyId"].astype(int)
        table = sanitize_numeric_frame(table.drop_duplicates("surveyId"))
        if result is None:
            result = table
        else:
            result = result.merge(table, on="surveyId", how="left")
    return result


def standardize_with_stats(arr, means, stds):
    """Standardize array using pre-computed stats."""
    return ((arr - means) / stds).astype(np.float32)


# ============================================================================
# PHASE 1: PO PRE-TRAINING DATA
# ============================================================================

def assign_region_country_to_po(po_df, pa_metadata_path):
    """Assign region and country to PO observations using nearest PA survey."""
    from scipy.spatial import cKDTree

    print("  Assigning region/country to PO via nearest PA survey...")
    pa_meta = pd.read_csv(pa_metadata_path)
    pa_unique = pa_meta.drop_duplicates("surveyId")[["lat", "lon", "region", "country"]].dropna()

    # Build KD-tree of PA locations
    pa_coords = np.deg2rad(pa_unique[["lat", "lon"]].values)
    pa_tree = cKDTree(pa_coords)

    po_coords = np.deg2rad(po_df[["lat", "lon"]].values)
    _, indices = pa_tree.query(po_coords, k=1)

    po_df = po_df.copy()
    po_df["region"] = pa_unique["region"].values[indices]
    po_df["country"] = pa_unique["country"].values[indices]

    print(f"    Assigned {po_df['country'].nunique()} countries, {po_df['region'].nunique()} regions")
    return po_df


def prepare_po_data(pa_species_ids, pa_data):
    """Prepare PO data with FULL 57-dim aux features (matching PA exactly)."""
    print("\n" + "=" * 60)
    print("PREPARING PO DATA FOR PRE-TRAINING (v7 - full aux)")
    print("=" * 60)

    cache_path = (Path(os.environ.get("GLC_OUTPUT_DIR", "artifacts")) / "artifacts_v7/cache/po_pretrain_v7.pkl")
    if cache_path.exists():
        print(f"  Loading cached PO data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    t0 = time.time()

    # Load PO metadata
    print("  Loading PO metadata...")
    po_meta = pd.read_csv(DATA_DIR / "GLC25_P0_metadata_train.csv")
    po_meta = po_meta.dropna(subset=["lat", "lon", "speciesId"])
    po_meta["speciesId"] = po_meta["speciesId"].astype(int)
    po_meta["surveyId"] = po_meta["surveyId"].astype(int)
    print(f"  PO total: {len(po_meta)} rows, {po_meta.speciesId.nunique()} species")

    # Filter to PA-vocabulary species only
    pa_species_set = set(int(x) for x in pa_species_ids)
    po_meta = po_meta[po_meta["speciesId"].isin(pa_species_set)].copy()
    print(f"  After PA-species filter: {len(po_meta)} rows, {po_meta.speciesId.nunique()} species")

    # Deduplicate by surveyId
    po_unique = po_meta.drop_duplicates("surveyId")
    print(f"  Unique surveys: {len(po_unique)}")

    # Load PO environmental data
    print("  Loading PO environmental features...")
    po_env = load_env_table("PO", "train")
    print(f"  PO env: {po_env.shape}")

    # Merge env
    po_merged = po_unique[["surveyId", "lat", "lon", "year", "speciesId", "geoUncertaintyInM"]].merge(
        po_env, on="surveyId", how="inner"
    )
    print(f"  After env merge: {len(po_merged)} surveys")

    # Assign region/country from nearest PA survey
    po_merged = assign_region_country_to_po(
        po_merged, DATA_DIR / "GLC25_PA_metadata_train.csv"
    )

    # Build env features (same columns as PA)
    env_cols = [c for c in po_merged.columns
                if c not in ["surveyId", "lat", "lon", "year", "speciesId",
                            "geoUncertaintyInM", "region", "country"]]
    po_env_arr = po_merged[env_cols].values.astype(np.float32)
    po_env_arr = np.nan_to_num(po_env_arr, nan=0.0)

    # Build aux features - SAME format as PA (will be standardized jointly)
    # PA aux = [lon, lat, year, geoUncertaintyInM, areaInM2, region_onehot, country_onehot]
    po_aux_df = po_merged[["lon", "lat", "year", "geoUncertaintyInM", "region", "country"]].copy()
    po_aux_df["areaInM2"] = np.nan  # PO doesn't have areaInM2
    po_aux_df["geoUncertaintyInM"] = po_aux_df["geoUncertaintyInM"].fillna(
        po_aux_df["geoUncertaintyInM"].median()
    )

    # Build PA aux dataframe for joint standardization
    meta_train = pa_data.metadata_train
    if isinstance(meta_train, np.ndarray):
        pa_meta_df = pd.DataFrame(meta_train,
                                   columns=["surveyId", "lon", "lat", "year", "region", "country"])
    else:
        pa_meta_df = meta_train.copy()

    # Load full PA metadata to get geoUncertainty and areaInM2
    pa_full = pd.read_csv(DATA_DIR / "GLC25_PA_metadata_train.csv")
    pa_full = pa_full.drop_duplicates("surveyId")
    pa_aux_df = pa_full[["lon", "lat", "year", "geoUncertaintyInM", "areaInM2", "region", "country"]].copy()

    # Also load test metadata for consistent one-hot encoding
    pa_test = pd.read_csv(DATA_DIR / "GLC25_PA_metadata_test.csv")
    pa_test = pa_test.drop_duplicates("surveyId")
    pa_test_aux_df = pa_test[["lon", "lat", "year", "geoUncertaintyInM", "areaInM2", "region", "country"]].copy()

    # Joint one-hot encoding: PA_train + PA_test + PO (to ensure all categories present)
    po_aux_df["_source"] = "po"
    pa_aux_df["_source"] = "pa_train"
    pa_test_aux_df["_source"] = "pa_test"
    combined = pd.concat([pa_aux_df, pa_test_aux_df, po_aux_df], axis=0, ignore_index=True)

    # Fill NaN
    for col in ["region", "country"]:
        combined[col] = combined[col].fillna("Unknown")
    combined["areaInM2"] = pd.to_numeric(combined["areaInM2"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # One-hot encode
    combined = pd.get_dummies(combined, columns=["region", "country"], dtype=np.float32)

    # Convert to numeric
    for col in combined.columns:
        if col != "_source":
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Split back
    pa_train_mask = combined["_source"] == "pa_train"
    pa_test_mask = combined["_source"] == "pa_test"
    po_mask = combined["_source"] == "po"
    combined = combined.drop(columns=["_source"])

    pa_train_arr = combined[pa_train_mask].values.astype(np.float32)
    po_arr = combined[po_mask].values.astype(np.float32)

    # Standardize using PA train stats (same normalization for PO)
    medians = np.nanmedian(pa_train_arr, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    pa_train_arr = np.where(np.isnan(pa_train_arr), medians, pa_train_arr)
    po_arr = np.where(np.isnan(po_arr), medians, po_arr)

    means = pa_train_arr.mean(axis=0)
    stds = np.where(pa_train_arr.std(axis=0) < 1e-6, 1.0, pa_train_arr.std(axis=0))

    po_aux_final = ((po_arr - means) / stds).astype(np.float32)

    # Standardize env using PA train stats too
    pa_env_train = pa_data.env_train  # already standardized
    # Re-standardize PO env using PA train stats for consistency
    env_means = po_env_arr.mean(axis=0)
    env_stds = np.where(po_env_arr.std(axis=0) < 1e-6, 1.0, po_env_arr.std(axis=0))
    po_env_final = ((po_env_arr - env_means) / env_stds).astype(np.float32)

    # Species labels
    sp_to_idx = {int(sp): i for i, sp in enumerate(pa_species_ids)}
    po_labels = np.array([sp_to_idx[int(sp)] for sp in po_merged["speciesId"].values], dtype=np.int64)

    print(f"  PO aux dim: {po_aux_final.shape[1]} (should match PA aux dim: {pa_data.aux_train.shape[1]})")

    result = {
        "env": po_env_final,
        "aux": po_aux_final,
        "labels": po_labels,
        "env_dim": po_env_final.shape[1],
        "aux_dim": po_aux_final.shape[1],
        "n_samples": len(po_labels),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  PO data ready: {result['n_samples']} samples, "
          f"env={result['env_dim']}d, aux={result['aux_dim']}d ({time.time()-t0:.1f}s)")
    print(f"  Cached to {cache_path}")
    return result


class PODataset(Dataset):
    """Simple dataset for PO pre-training (env + aux -> species)."""
    def __init__(self, env, aux, labels):
        self.env = torch.tensor(env, dtype=torch.float32)
        self.aux = torch.tensor(aux, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.env[idx], self.aux[idx], self.labels[idx]


# ============================================================================
# PO PRE-TRAINING MODEL (env + aux only)
# ============================================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dims=(256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class POPretrainModel(nn.Module):
    """Simplified model for PO pre-training: env_encoder + aux_encoder -> classifier."""
    def __init__(self, n_species, env_dim, aux_dim, branch_dim=128):
        super().__init__()
        self.environment_encoder = MLPEncoder(env_dim, embedding_dim=branch_dim, dropout=0.1)
        self.aux_encoder = MLPEncoder(aux_dim, embedding_dim=branch_dim, hidden_dims=(128, 128), dropout=0.1)
        # Classifier head for cross-entropy (single-label PO)
        self.head = nn.Sequential(
            nn.Linear(2 * branch_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, n_species),
        )

    def forward(self, env, aux):
        env_emb = self.environment_encoder(env)
        aux_emb = self.aux_encoder(aux)
        return self.head(torch.cat([env_emb, aux_emb], dim=1))


def pretrain_on_po(po_data, n_species, branch_dim=128, epochs=5, lr=1e-3, batch_size=512):
    """Phase 1: Pre-train env+aux encoders on PO data."""
    print("\n" + "=" * 60)
    print("PHASE 1: PO PRE-TRAINING")
    print("=" * 60)

    # Split PO into train/val (95/5)
    n = po_data["n_samples"]
    indices = np.random.permutation(n)
    val_size = min(50000, n // 20)  # 5% or max 50K
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_ds = PODataset(po_data["env"][train_idx], po_data["aux"][train_idx], po_data["labels"][train_idx])
    val_ds = PODataset(po_data["env"][val_idx], po_data["aux"][val_idx], po_data["labels"][val_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  Env dim: {po_data['env_dim']}, Aux dim: {po_data['aux_dim']}")

    model = POPretrainModel(n_species, po_data["env_dim"], po_data["aux_dim"], branch_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0
        correct = 0
        total = 0

        for env, aux, labels in train_dl:
            env = env.to(DEVICE, non_blocking=True)
            aux = aux.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(env, aux)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for env, aux, labels in val_dl:
                env = env.to(DEVICE, non_blocking=True)
                aux = aux.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                logits = model(env, aux)
                val_loss += criterion(logits, labels).item()
                val_batches += 1
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        train_acc = correct / total
        val_acc = val_correct / val_total
        dt = time.time() - t0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch+1}/{epochs} | "
              f"TrLoss={total_loss/n_batches:.4f} TrAcc={train_acc:.4f} | "
              f"VLoss={val_loss/val_batches:.4f} VAcc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} {dt:.0f}s"
              f"{' *** BEST' if is_best else ''}")

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    print(f"  (Random baseline: {1/n_species:.6f})")

    return best_state, po_data["env_dim"], po_data["aux_dim"]


# ============================================================================
# PHASE 2: PA FINE-TUNING (reuse train.py infrastructure)
# ============================================================================
# We import the full training pipeline from train.py but inject pre-trained weights

@dataclass
class PreparedData:
    env_train: np.ndarray = None
    env_test: np.ndarray = None
    aux_train: np.ndarray = None
    aux_test: np.ndarray = None
    landsat_train_paths: list = field(default_factory=list)
    landsat_test_paths: list = field(default_factory=list)
    climate_train_paths: list = field(default_factory=list)
    climate_test_paths: list = field(default_factory=list)
    image_train_paths: list = field(default_factory=list)
    image_test_paths: list = field(default_factory=list)
    train_labels: list = field(default_factory=list)
    species_ids: np.ndarray = None
    train_survey_ids: np.ndarray = None
    test_survey_ids: np.ndarray = None
    metadata_train: np.ndarray = None
    has_test: bool = True


# Import model components (same as model.py)
class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, in_features, embedding_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(in_features, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=embedding_dim*4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        return self.norm(x).mean(dim=1)


class FusionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.classifier(self.fusion(x))


class MultimodalPlantModel(nn.Module):
    def __init__(self, n_species, env_dim, aux_dim, branch_dim=128,
                 fusion_hidden_dim=512, dropout=0.2, use_image_branch=False):
        super().__init__()
        self.use_image_branch = use_image_branch
        self.image_encoder = None
        self.landsat_encoder = TimeSeriesTransformerEncoder(24, embedding_dim=branch_dim, dropout=dropout)
        self.climate_encoder = TimeSeriesTransformerEncoder(76, embedding_dim=branch_dim, dropout=dropout)
        self.environment_encoder = MLPEncoder(env_dim, embedding_dim=branch_dim, dropout=dropout)
        self.aux_encoder = MLPEncoder(aux_dim, embedding_dim=branch_dim, hidden_dims=(128, 128), dropout=dropout)
        n_branches = 4 + int(use_image_branch)
        self.head = FusionHead(n_branches * branch_dim, n_species, hidden_dim=fusion_hidden_dim, dropout=dropout)

    def forward(self, env, aux, landsat, climate, image=None):
        embeddings = [
            self.environment_encoder(env),
            self.aux_encoder(aux),
            self.landsat_encoder(landsat),
            self.climate_encoder(climate),
        ]
        if self.use_image_branch and self.image_encoder is not None:
            if image is None:
                raise ValueError("Image branch enabled but no image")
            embeddings.insert(0, self.image_encoder(image))
        return self.head(torch.cat(embeddings, dim=1))


def load_tensor_cube(path):
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError, OSError, ValueError, pickle.UnpicklingError):
        return None
    if isinstance(tensor, dict):
        tensor = next((v for v in tensor.values() if isinstance(v, torch.Tensor)), None)
    if not isinstance(tensor, torch.Tensor):
        return None
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


class SurveyDataset(Dataset):
    def __init__(self, env, aux, landsat_paths, climate_paths, labels, n_classes):
        self.env = env
        self.aux = aux
        self.landsat_paths = landsat_paths
        self.climate_paths = climate_paths
        self.labels = labels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.env)

    @staticmethod
    def _normalize_tensor(t):
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
        return {
            "env": torch.from_numpy(self.env[idx]).float(),
            "aux": torch.from_numpy(self.aux[idx]).float(),
            "landsat": landsat,
            "climate": climate,
            "labels": None if self.labels is None else self.labels[idx],
        }


def collate_fn(batch, n_classes):
    env = torch.stack([b["env"] for b in batch])
    aux = torch.stack([b["aux"] for b in batch])
    landsat = torch.stack([b["landsat"] for b in batch])
    climate = torch.stack([b["climate"] for b in batch])
    targets = None
    if batch[0]["labels"] is not None:
        targets = torch.zeros((len(batch), n_classes), dtype=torch.float32)
        for i, b in enumerate(batch):
            targets[i, b["labels"]] = 1.0
    return env, aux, landsat, climate, targets


def compute_f1_score(preds, targets):
    """Compute macro-averaged F1 score."""
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]


def find_best_alpha(probs, labels_list, n_classes, alpha_grid=ALPHA_GRID):
    """Find best alpha for adaptive thresholding."""
    best_alpha, best_f1 = 1.0, 0.0
    targets = torch.zeros(len(labels_list), n_classes)
    for i, labs in enumerate(labels_list):
        targets[i, labs] = 1.0

    for alpha in alpha_grid:
        preds = torch.zeros_like(targets)
        for i in range(len(probs)):
            p = probs[i]
            k = int(np.clip(np.round(p.sum().item() * alpha), 1, 50))
            topk = torch.topk(p, k).indices
            preds[i, topk] = 1.0
        f1 = compute_f1_score(preds, targets)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
    return best_alpha, best_f1


def train_pa_fold(model, train_ds, val_ds, n_classes, fold_idx, epochs=20, lr=5e-4, output_dir=None):
    """Train one fold of PA fine-tuning."""
    train_dl = DataLoader(train_ds, batch_size=PA_BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True,
                          collate_fn=partial(collate_fn, n_classes=n_classes))
    val_dl = DataLoader(val_ds, batch_size=PA_BATCH_SIZE * 2, shuffle=False,
                        num_workers=0, pin_memory=True,
                        collate_fn=partial(collate_fn, n_classes=n_classes))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    # Warmup
    warmup_epochs = 2
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=warmup_epochs)

    best_f1 = 0
    best_alpha = 1.4
    best_epoch = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0

        for env, aux, landsat, climate, targets in train_dl:
            env = env.to(DEVICE, non_blocking=True)
            aux = aux.to(DEVICE, non_blocking=True)
            landsat = landsat.to(DEVICE, non_blocking=True)
            climate = climate.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            logits = model(env, aux, landsat, climate)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for env, aux, landsat, climate, targets in val_dl:
                env = env.to(DEVICE, non_blocking=True)
                aux = aux.to(DEVICE, non_blocking=True)
                landsat = landsat.to(DEVICE, non_blocking=True)
                climate = climate.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                logits = model(env, aux, landsat, climate)
                val_loss += nn.BCEWithLogitsLoss()(logits, targets).item()
                val_batches += 1
                all_probs.append(torch.sigmoid(logits).cpu())

        probs = torch.cat(all_probs, dim=0)
        alpha, f1 = find_best_alpha(probs, val_ds.labels, n_classes)
        dt = time.time() - t0

        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            best_alpha = alpha
            best_epoch = epoch + 1
            if output_dir:
                torch.save({
                    "state_dict": model.state_dict(),
                    "fold": fold_idx,
                    "epoch": epoch + 1,
                    "alpha": alpha,
                    "score": f1,
                    "n_classes": n_classes,
                    "env_dim": model.environment_encoder.network[0].in_features,
                    "aux_dim": model.aux_encoder.network[0].in_features,
                    "branch_dim": BRANCH_DIM,
                    "fusion_hidden_dim": FUSION_HIDDEN_DIM,
                    "dropout": 0.2,
                    "use_image_branch": False,
                }, output_dir / f"fold{fold_idx}_best.pt")

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Fold{fold_idx} Ep {epoch+1:2d}/{epochs} | "
              f"TrLoss={total_loss/n_batches:.5f} VLoss={val_loss/val_batches:.5f} "
              f"F1={f1:.5f} alpha={alpha:.2f} lr={lr_now:.6f} {dt:.0f}s"
              f"{' *** BEST' if is_best else ''}")

    return best_f1, best_alpha, best_epoch


def inject_pretrained_weights(model, pretrained_state):
    """
    Transfer pre-trained env/aux encoder weights from PO model to PA model.
    v7: dimensions match exactly, so full transfer.
    """
    model_state = model.state_dict()
    transferred = 0
    skipped = 0

    for key in pretrained_state:
        if not (key.startswith("environment_encoder.") or key.startswith("aux_encoder.")):
            continue
        if key in model_state and pretrained_state[key].shape == model_state[key].shape:
            model_state[key] = pretrained_state[key]
            transferred += 1
        else:
            skipped += 1
            if key in model_state:
                print(f"    SKIP (shape mismatch): {key} "
                      f"{pretrained_state[key].shape} vs {model_state[key].shape}")

    model.load_state_dict(model_state)
    print(f"  Transferred {transferred} params, skipped {skipped}")
    return model


# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================

@torch.no_grad()
def ensemble_predict(models_info, test_ds, n_classes, species_ids, test_survey_ids, output_dir):
    """Generate ensemble predictions from multiple folds."""
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0,
                         pin_memory=True, collate_fn=partial(collate_fn, n_classes=n_classes))

    all_logits = []
    avg_alpha = 0

    for fold_info in models_info:
        ckpt = torch.load(fold_info["path"], map_location="cpu")
        state = ckpt["state_dict"]
        alpha = ckpt["alpha"]
        f1 = ckpt["score"]
        env_dim = ckpt["env_dim"]
        aux_dim = ckpt["aux_dim"]

        model = MultimodalPlantModel(n_classes, env_dim, aux_dim, branch_dim=BRANCH_DIM,
                                     fusion_hidden_dim=FUSION_HIDDEN_DIM)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()

        fold_logits = []
        for env, aux, landsat, climate, _ in test_dl:
            env = env.to(DEVICE, non_blocking=True)
            aux = aux.to(DEVICE, non_blocking=True)
            landsat = landsat.to(DEVICE, non_blocking=True)
            climate = climate.to(DEVICE, non_blocking=True)
            logits = model(env, aux, landsat, climate)
            fold_logits.append(logits.cpu())

        all_logits.append(torch.cat(fold_logits, 0))
        avg_alpha += alpha
        print(f"  Loaded fold {fold_info['fold']}: F1={f1:.5f}, alpha={alpha:.2f}")
        del model
        torch.cuda.empty_cache()

    avg_alpha /= len(models_info)
    avg_logits = torch.stack(all_logits).mean(0)
    probs = torch.sigmoid(avg_logits)

    print(f"\n  Ensemble: {len(models_info)} folds, avg_alpha={avg_alpha:.2f}")

    # Generate submissions for various alphas
    alpha_grid = [avg_alpha, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
    alpha_grid = sorted(set(alpha_grid))

    for alpha in alpha_grid:
        preds_list = []
        for i in range(len(probs)):
            p = probs[i]
            k = int(np.clip(np.round(p.sum().item() * alpha), 1, 50))
            topk = torch.topk(p, k).indices
            species = sorted([int(species_ids[j]) for j in topk])
            preds_list.append(" ".join(map(str, species)))

        sub = pd.DataFrame({"surveyId": test_survey_ids, "predictions": preds_list})
        fname = f"submission_ens{len(models_info)}_alpha{alpha:.1f}.csv"
        sub.to_csv(output_dir / fname, index=False)
        counts = sub["predictions"].apply(lambda x: len(str(x).split()))
        print(f"  -> {fname} (median={counts.median():.0f}, mean={counts.mean():.1f})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("V7: PO PRE-TRAINING + PA FINE-TUNING (full aux)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(SEED)

    output_dir = (Path(os.environ.get("GLC_OUTPUT_DIR", "artifacts")) / "artifacts_v7")
    output_dir.mkdir(exist_ok=True)

    # ================================================================
    # Load PA data (reuse existing cache)
    # ================================================================
    print("\n[1] Loading PA data...")
    cache_path = (Path(os.environ.get("GLC_OUTPUT_DIR", "artifacts")) / "artifacts/cache/prepared_multimodal.pkl")
    sys.modules['__main__'].PreparedData = PreparedData
    with open(cache_path, "rb") as f:
        pa_data = pickle.load(f)

    n_species = len(pa_data.species_ids)
    env_dim = pa_data.env_train.shape[1]
    aux_dim = pa_data.aux_train.shape[1]
    print(f"  PA: {len(pa_data.train_survey_ids)} surveys, {n_species} species")
    print(f"  Env: {env_dim}d, Aux: {aux_dim}d")

    # ================================================================
    # Phase 1: PO Pre-training
    # ================================================================
    print("\n[2] Preparing PO data...")
    po_data = prepare_po_data(pa_data.species_ids, pa_data)

    pretrained_state, po_env_dim, po_aux_dim = pretrain_on_po(
        po_data, n_species,
        branch_dim=BRANCH_DIM,
        epochs=PO_PRETRAIN_EPOCHS,
        lr=PO_PRETRAIN_LR,
        batch_size=PO_BATCH_SIZE,
    )

    # Save PO pretrained weights
    torch.save(pretrained_state, output_dir / "po_pretrained.pt")
    print(f"  Saved PO pretrained weights to {output_dir}/po_pretrained.pt")

    # ================================================================
    # Phase 2: PA Fine-tuning with pretrained weights
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: PA FINE-TUNING (with PO pre-trained weights)")
    print("=" * 60)

    # Build spatial blocks for cross-validation
    meta_train = pa_data.metadata_train
    if isinstance(meta_train, np.ndarray):
        meta_df = pd.DataFrame(meta_train, columns=["surveyId", "lon", "lat", "year", "region", "country"])
    else:
        meta_df = meta_train

    lat = meta_df["lat"].astype(float).values
    lon = meta_df["lon"].astype(float).values
    lat_bins = np.floor(lat / 1.0).astype(np.int32)
    lon_bins = np.floor(lon / 1.0).astype(np.int32)
    groups = np.char.add(np.char.add(lat_bins.astype(str), "_"), lon_bins.astype(str))

    # Build fold masks
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    fold_groups = np.array_split(unique_groups, N_FOLDS)
    fold_masks = [np.isin(groups, fg) for fg in fold_groups]

    print(f"\n[TRAINING ALL {N_FOLDS} FOLDS]")

    fold_results = []
    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx} / {N_FOLDS - 1}")
        val_mask = fold_masks[fold_idx]
        train_mask = ~val_mask
        print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}")
        print("=" * 60)

        # Create datasets
        train_ds = SurveyDataset(
            pa_data.env_train[train_mask], pa_data.aux_train[train_mask],
            [pa_data.landsat_train_paths[i] for i in np.where(train_mask)[0]],
            [pa_data.climate_train_paths[i] for i in np.where(train_mask)[0]],
            [pa_data.train_labels[i] for i in np.where(train_mask)[0]],
            n_species)
        val_ds = SurveyDataset(
            pa_data.env_train[val_mask], pa_data.aux_train[val_mask],
            [pa_data.landsat_train_paths[i] for i in np.where(val_mask)[0]],
            [pa_data.climate_train_paths[i] for i in np.where(val_mask)[0]],
            [pa_data.train_labels[i] for i in np.where(val_mask)[0]],
            n_species)

        # Create model and inject pre-trained weights
        model = MultimodalPlantModel(n_species, env_dim, aux_dim,
                                     branch_dim=BRANCH_DIM, fusion_hidden_dim=FUSION_HIDDEN_DIM)

        inject_pretrained_weights(model, pretrained_state)
        model.to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        best_f1, best_alpha, best_epoch = train_pa_fold(
            model, train_ds, val_ds, n_species, fold_idx,
            epochs=PA_EPOCHS, lr=PA_LR, output_dir=output_dir)

        print(f"\n  Fold {fold_idx} DONE: best F1={best_f1:.5f} at epoch {best_epoch}, alpha={best_alpha:.2f}")
        fold_results.append({
            "fold": fold_idx,
            "f1": best_f1,
            "alpha": best_alpha,
            "epoch": best_epoch,
            "path": output_dir / f"fold{fold_idx}_best.pt",
        })

        del model
        torch.cuda.empty_cache()

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("RESUME ALL FOLDS")
    print("=" * 60)
    for r in fold_results:
        print(f"  Fold {r['fold']}: F1={r['f1']:.5f}, alpha={r['alpha']:.2f}")
    avg_f1 = np.mean([r["f1"] for r in fold_results])
    print(f"  Average F1: {avg_f1:.5f}")

    # ================================================================
    # Ensemble prediction
    # ================================================================
    print("\n[AUTO ENSEMBLE PREDICT]")

    test_ds = SurveyDataset(
        pa_data.env_test, pa_data.aux_test,
        pa_data.landsat_test_paths, pa_data.climate_test_paths,
        None, n_species)

    ensemble_predict(fold_results, test_ds, n_species,
                     pa_data.species_ids, pa_data.test_survey_ids, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
