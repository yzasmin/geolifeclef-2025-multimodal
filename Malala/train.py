# -*- coding: utf-8 -*-
"""
Baseline - GeoLifeCLEF 2025 (PA only)
======================================

Split spatial : les surveys sont divises en blocs lat/lon de taille `--block-deg`
degres (defaut 2°). ~15% des blocs (1 sur 7) sont reserves pour la validation.
Cela teste la generalisation geographique reelle, pas juste l'interpolation.

Usage:
    python3 train.py --explore
    python3 train.py --epochs 30 --batch-size 64 --gpu-id 0
    python3 train.py --predict --checkpoint checkpoints/best.pt --gpu-id 0

Donnees:
    /data/challenge2026MIASHS/PA/PA-train/
    /data/challenge2026MIASHS/PA/PA-test/
    /data/challenge2026MIASHS/EnvironmentalValues/
    ~/GLC25_PA_metadata_train.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ============================================================================
# CHEMINS
# ============================================================================

BASE_DIR = "/data/challenge2026MIASHS"

PATHS = {
    "pa_train":        os.path.join(BASE_DIR, "PA", "PA-train"),
    "pa_test":         os.path.join(BASE_DIR, "PA", "PA-test"),
    "bioclim_train":   os.path.join(BASE_DIR, "EnvironmentalValues", "ClimateAverage_1981-2010", "GLC25-PA-train-bioclimatic.csv"),
    "bioclim_test":    os.path.join(BASE_DIR, "EnvironmentalValues", "ClimateAverage_1981-2010", "GLC25-PA-test-bioclimatic.csv"),
    "landcover_train": os.path.join(BASE_DIR, "EnvironmentalValues", "LandCover", "GLC25-PA-train-landcover.csv"),
    "landcover_test":  os.path.join(BASE_DIR, "EnvironmentalValues", "LandCover", "GLC25-PA-test-landcover.csv"),
    "soilgrids_train": os.path.join(BASE_DIR, "EnvironmentalValues", "SoilGrids", "GLC25-PA-train-soilgrids.csv"),
    "soilgrids_test":  os.path.join(BASE_DIR, "EnvironmentalValues", "SoilGrids", "GLC25-PA-test-soilgrids.csv"),
    "human_fp_train":  os.path.join(BASE_DIR, "EnvironmentalValues", "HumanFootprint", "GLC25-PA-train-human_footprint.csv"),
    "human_fp_test":   os.path.join(BASE_DIR, "EnvironmentalValues", "HumanFootprint", "GLC25-PA-test-human_footprint.csv"),
    "elevation_train": os.path.join(BASE_DIR, "EnvironmentalValues", "Elevation", "GLC25-PA-train-elevation.csv"),
    "elevation_test":  os.path.join(BASE_DIR, "EnvironmentalValues", "Elevation", "GLC25-PA-test-elevation.csv"),
}

LABELS_CSV = os.path.expanduser("~/GLC25_PA_metadata_train.csv")


# ============================================================================
# CHARGEMENT DES FEATURES ENVIRONNEMENTALES
# ============================================================================

def load_csv_as_dict(csv_path):
    # type: (str) -> Tuple[Dict[int, np.ndarray], List[str]]
    """Charge un CSV env en {surveyId: np.array}. 1ere colonne = surveyId."""
    data = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            try:
                sid  = int(row[0])
                vals = [float(x) if x.strip() != "" else 0.0 for x in row[1:]]
                data[sid] = np.array(vals, dtype=np.float32)
            except (ValueError, IndexError):
                continue
    return data, header[1:]


def load_all_env_features(split="train"):
    # type: (str) -> Tuple[Dict[int, np.ndarray], List[str]]
    """Charge et concatene toutes les features CSV pour un split."""
    suffix   = "_" + split
    csv_keys = [k for k in PATHS if k.endswith(suffix) and not k.startswith("pa")]

    all_data  = {}   # type: Dict[int, List[np.ndarray]]
    all_names = []   # type: List[str]

    for key in sorted(csv_keys):
        path = PATHS[key]
        if not os.path.exists(path):
            print("  WARN: {} introuvable".format(path))
            continue
        data, names = load_csv_as_dict(path)
        print("  {}: {} surveys, {} features".format(
            os.path.basename(path), len(data), len(names)))
        all_names.extend(names)
        for sid, vals in data.items():
            all_data.setdefault(sid, []).append(vals)

    n_feats = len(all_names)
    result  = {sid: np.concatenate(parts)
               for sid, parts in all_data.items()
               if len(np.concatenate(parts)) == n_feats}

    print("  Total: {} surveys, {} features".format(len(result), n_feats))
    return result, all_names


def compute_norm_stats(env_data):
    # type: (Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]
    """Z-score : mean et std sur l'ensemble d'entrainement."""
    matrix = np.stack(list(env_data.values()))
    mean   = matrix.mean(axis=0)
    std    = matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


# ============================================================================
# CHARGEMENT DES LABELS ET COORDONNEES
# ============================================================================

def load_labels_and_coords(labels_csv):
    # type: (str) -> Tuple[Dict[int, set], List[int], Dict[int, Tuple[float, float]]]
    """
    Charge le CSV metadata.
    Retourne :
      labels  : {surveyId: {spId, ...}}
      species : liste triee des especes uniques
      coords  : {surveyId: (lon, lat)}  - pour le split spatial
    """
    if not os.path.exists(labels_csv):
        print("ERREUR : {} introuvable".format(labels_csv))
        sys.exit(1)

    labels  = defaultdict(set)   # type: Dict[int, set]
    coords  = {}                 # type: Dict[int, Tuple[float, float]]
    species = set()

    with open(labels_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        sid_col = header.index("surveyid")
        sp_col  = header.index("speciesid")
        lon_col = header.index("lon")
        lat_col = header.index("lat")

        for row in reader:
            try:
                sid = int(row[sid_col])
                sp  = int(float(row[sp_col]))
                lon = float(row[lon_col])
                lat = float(row[lat_col])
                labels[sid].add(sp)
                species.add(sp)
                coords[sid] = (lon, lat)   # derniere valeur = identique par survey
            except (ValueError, IndexError):
                continue

    counts = [len(v) for v in labels.values()]
    print("  {} surveys | {} especes | esp/survey moy={:.1f}".format(
        len(labels), len(species), sum(counts) / len(counts)))
    return dict(labels), sorted(species), coords


# ============================================================================
# SPLIT SPATIAL
# ============================================================================

def spatial_split(survey_ids, coords, block_deg=2.0, val_every=7, seed=42):
    # type: (List[int], Dict[int, Tuple[float, float]], float, int, int) -> Tuple[List[int], List[int]]
    """
    Divise les surveys en train/val selon un decoupage geographique.

    Principe :
      1. Chaque survey est assigne a une cellule de grille (lon//block, lat//block).
      2. Les cellules sont tries par (lat_block, lon_block).
      3. 1 cellule sur `val_every` va en validation (~14% si val_every=7).

    Avantage par rapport au split aleatoire :
      Les points proches dans l'espace ont tendance a avoir les memes especes
      (autocorrelation spatiale). Un split aleatoire entrainerait une fuite
      d'information entre train et val. Le split spatial teste si le modele
      generalise a de nouvelles zones geographiques.

    Args:
        block_deg : taille d'un bloc en degres (2° ≈ 200 km)
        val_every : 1 bloc sur val_every -> validation
    """
    # Assigner chaque survey a une cellule
    sid_to_cell = {}
    for sid in survey_ids:
        if sid in coords:
            lon, lat = coords[sid]
            cell = (int(lon // block_deg), int(lat // block_deg))
        else:
            cell = (0, 0)
        sid_to_cell[sid] = cell

    # Lister les cellules uniques et les trier
    all_cells = sorted(set(sid_to_cell.values()))

    # Choisir les cellules de validation (1 sur val_every)
    # On utilise seed pour reproducibilite
    rng = list(range(len(all_cells)))
    val_cell_indices = set(rng[seed % val_every::val_every])
    val_cells  = {all_cells[i] for i in val_cell_indices}
    train_cells = set(all_cells) - val_cells

    train_sids = [sid for sid in survey_ids if sid_to_cell[sid] in train_cells]
    val_sids   = [sid for sid in survey_ids if sid_to_cell[sid] in val_cells]

    print("  Blocs totaux: {} | train: {} | val: {}".format(
        len(all_cells), len(train_cells), len(val_cells)))
    print("  Surveys -> train: {} | val: {} ({:.1f}%)".format(
        len(train_sids), len(val_sids),
        100.0 * len(val_sids) / max(1, len(survey_ids))))

    return train_sids, val_sids


# ============================================================================
# CHARGEMENT DES IMAGES SENTINEL-2
# ============================================================================

def get_tiff_path(survey_id, split="train"):
    # type: (int, str) -> str
    """Convention de chemin : .../CD/AB/XXXXABCD.tiff"""
    base = PATHS["pa_" + split]
    h1   = "{:02d}".format(survey_id % 100)
    h2   = "{:02d}".format((survey_id // 100) % 100)
    return os.path.join(base, h1, h2, "{}.tiff".format(survey_id))


def load_sentinel_tiff(path):
    # type: (str) -> Optional[np.ndarray]
    """
    Charge un TIFF Sentinel-2, retourne float32 [4, 64, 64] normalise [0, 1].

    Notes sur les images :
    - 4 bandes : R, G, B, NIR (reflectance de surface)
    - Valeurs brutes en uint16, reflectance = valeur / 10000
    - Resolution : 10 m/pixel → patch 640m × 640m
    - Les images sont deja pre-traitees (atmospherique) par Ecodatacube
    """
    if not os.path.exists(path):
        return None
    try:
        import tifffile
        img = tifffile.imread(path)
    except Exception:
        return None

    if img.ndim == 2:
        img = img[np.newaxis]
    elif img.ndim == 3 and img.shape[2] <= img.shape[0]:
        img = np.transpose(img, (2, 0, 1))   # HWC -> CHW

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 10000.0
    img = np.clip(img, 0.0, 1.0)

    C, H, W = img.shape
    if H != 64 or W != 64:
        try:
            from PIL import Image as PILImage
            resized = []
            for c in range(C):
                band = PILImage.fromarray(img[c])
                resized.append(np.array(band.resize((64, 64), PILImage.BILINEAR)))
            img = np.stack(resized)
        except Exception:
            img = img[:, :64, :64]

    if img.shape[0] < 4:
        pad = np.zeros((4 - img.shape[0], 64, 64), dtype=np.float32)
        img = np.concatenate([img, pad], axis=0)

    return img[:4]


# ============================================================================
# DATASET
# ============================================================================

def build_dataset_class():
    """Retourne la classe PADataset (import torch differe)."""
    import torch
    from torch.utils.data import Dataset

    class PADataset(Dataset):
        """
        Dataset Presence-Absence - baseline (Sentinel-2 + env scalaires).

        augment=True applique des flips et rotations aleatoires sur l'image
        Sentinel-2 (invariances naturelles pour les images satellites).
        """

        def __init__(
            self,
            survey_ids,         # type: List[int]
            env_data,           # type: Dict[int, np.ndarray]
            labels=None,        # type: Optional[Dict[int, set]]
            species_list=None,  # type: Optional[List[int]]
            split="train",
            env_mean=None,      # type: Optional[np.ndarray]
            env_std=None,       # type: Optional[np.ndarray]
            augment=False,
        ):
            self.survey_ids  = survey_ids
            self.env_data    = env_data
            self.labels      = labels
            self.split       = split
            self.env_mean    = env_mean
            self.env_std     = env_std
            self.augment     = augment
            self.n_classes   = len(species_list) if species_list else 0
            self.sp2idx      = {sp: i for i, sp in enumerate(species_list)} if species_list else {}
            self.n_env       = len(env_mean) if env_mean is not None else 52

        def __len__(self):
            return len(self.survey_ids)

        def __getitem__(self, idx):
            import random
            sid = self.survey_ids[idx]

            # -- Sentinel-2 --
            tiff = load_sentinel_tiff(get_tiff_path(sid, self.split))
            if tiff is None:
                tiff = np.zeros((4, 64, 64), dtype=np.float32)
            sentinel = torch.from_numpy(tiff)

            # Augmentation : flips H/V et rotations 90° (train seulement)
            # Ces transformations respectent la symetrie des images satellites
            if self.augment:
                if random.random() > 0.5:
                    sentinel = torch.flip(sentinel, dims=[2])   # flip horizontal
                if random.random() > 0.5:
                    sentinel = torch.flip(sentinel, dims=[1])   # flip vertical
                k = random.randint(0, 3)
                if k > 0:
                    sentinel = torch.rot90(sentinel, k, dims=[1, 2])

            # -- Features environnementales --
            if sid in self.env_data:
                env = self.env_data[sid].copy()
                if self.env_mean is not None:
                    env = (env - self.env_mean) / self.env_std
                env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                env = np.zeros(self.n_env, dtype=np.float32)
            env_t = torch.from_numpy(env.astype(np.float32))

            # -- Labels --
            if self.labels is not None and sid in self.labels:
                label_vec = np.zeros(self.n_classes, dtype=np.float32)
                for sp in self.labels[sid]:
                    if sp in self.sp2idx:
                        label_vec[self.sp2idx[sp]] = 1.0
                label = torch.from_numpy(label_vec)
            else:
                label = torch.zeros(self.n_classes, dtype=torch.float32)

            return {
                "survey_id": sid,
                "image":     sentinel,
                "env":       env_t,
                "label":     label,
            }

    return PADataset


# ============================================================================
# METRIQUES
# ============================================================================

def compute_f1(preds_bin, labels):
    # type: (torch.Tensor, torch.Tensor) -> float
    """F1 sample-averaged (metrique officielle Kaggle)."""
    tp = (preds_bin * labels).sum(dim=-1)
    fp = (preds_bin * (1 - labels)).sum(dim=-1)
    fn = ((1 - preds_bin) * labels).sum(dim=-1)
    return (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()


def find_best_threshold(probs, labels):
    # type: (torch.Tensor, torch.Tensor) -> Tuple[float, float]
    """Cherche le seuil optimal sur [0.05, 0.50] par pas de 0.05."""
    best_t, best_f1 = 0.2, 0.0
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        f1 = compute_f1((probs > t).float(), labels)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def compute_pos_weight(labels, species_list):
    # type: (Dict[int, set], List[int]) -> torch.Tensor
    """
    pos_weight[k] = n_negatifs[k] / n_positifs[k], plafonne a 100.
    Force le modele a accorder plus d'importance aux especes rares.
    """
    n       = len(labels)
    sp2idx  = {sp: i for i, sp in enumerate(species_list)}
    pos_cnt = torch.zeros(len(species_list))
    for sp_set in labels.values():
        for sp in sp_set:
            if sp in sp2idx:
                pos_cnt[sp2idx[sp]] += 1
    neg_cnt    = n - pos_cnt
    pos_weight = (neg_cnt / (pos_cnt + 1e-8)).clamp(max=100.0)
    print("  pos_weight : min={:.1f}  max={:.1f}  mean={:.1f}".format(
        pos_weight.min().item(), pos_weight.max().item(), pos_weight.mean().item()))
    return pos_weight


# ============================================================================
# ENTRAINEMENT
# ============================================================================

def run_train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from model import BaselineModel, setup_device

    device = setup_device(args.gpu_id)

    # 1. Labels + coordonnees
    print("\n[1/5] Labels...")
    labels, species_list, coords = load_labels_and_coords(args.labels)
    n_classes = len(species_list)
    print("  {} classes".format(n_classes))

    # 2. Features environnementales
    print("\n[2/5] Features environnementales...")
    env_data, env_names = load_all_env_features("train")
    env_mean, env_std   = compute_norm_stats(env_data)
    n_env = len(env_names)
    print("  {} features env".format(n_env))

    # 3. Split spatial
    print("\n[3/5] Split spatial (blocs {}°)...".format(args.block_deg))
    valid_sids = [sid for sid in labels if sid in env_data]
    train_sids, val_sids = spatial_split(
        valid_sids, coords,
        block_deg=args.block_deg,
        val_every=args.val_every,
    )

    # 4. Datasets
    print("\n[4/5] Datasets...")
    PADataset = build_dataset_class()
    train_ds  = PADataset(train_sids, env_data, labels=labels,
                          species_list=species_list, split="train",
                          env_mean=env_mean, env_std=env_std, augment=True)
    val_ds    = PADataset(val_sids,   env_data, labels=labels,
                          species_list=species_list, split="train",
                          env_mean=env_mean, env_std=env_std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    print("  Train: {}  Val: {}".format(len(train_ds), len(val_ds)))

    # 5. Modele
    print("\n[5/5] Modele...")
    model  = BaselineModel(n_species=n_classes, n_env=n_env, pretrained=True).to(device)
    n_pars = sum(p.numel() for p in model.parameters())
    print("  Parametres: {:,}".format(n_pars))

    # pos_weight pour compenser le desequilibre especes rares / communes
    print("  Calcul pos_weight...")
    pos_weight = compute_pos_weight(labels, species_list).to(device)

    # LR differentiel : backbone ResNet × 0.1
    backbone_params = list(model.sentinel_enc.parameters())
    other_params    = list(model.env_enc.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Mixed precision
    use_amp   = (str(device) != "cpu")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler    = torch.amp.GradScaler() if (use_amp and amp_dtype == torch.float16) else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1        = 0.0
    best_threshold = 0.2
    patience_cnt   = 0

    print("\n" + "=" * 65)
    print("Entrainement : {} epochs | batch {} | lr {} | AMP {}".format(
        args.epochs, args.batch_size, args.lr,
        str(amp_dtype).split(".")[-1] if use_amp else "off"))
    print("=" * 65 + "\n")

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        t_loss, n_b = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            env   = batch["env"].to(device,   non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):
                    logits = model(image=image, env=env)
                    loss   = criterion(logits, label)
            else:
                logits = model(image=image, env=env)
                loss   = criterion(logits, label)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            t_loss += loss.item()
            n_b    += 1

        scheduler.step()

        # ── Validation ──
        model.eval()
        v_loss, n_vb          = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                env   = batch["env"].to(device,   non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)

                if use_amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):
                        logits = model(image=image, env=env)
                        loss   = criterion(logits, label)
                else:
                    logits = model(image=image, env=env)
                    loss   = criterion(logits, label)

                v_loss += loss.item()
                n_vb   += 1
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(label.cpu())

        preds_cat  = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        best_t, f1 = find_best_threshold(preds_cat, labels_cat)

        print("Epoch {:3d}/{} | loss {:.4f} | val_loss {:.4f} | F1@{:.2f}={:.4f} | {:.0f}s".format(
            epoch + 1, args.epochs,
            t_loss / max(n_b, 1),
            v_loss / max(n_vb, 1),
            best_t, f1,
            time.time() - t0))

        # Sauvegarder le meilleur checkpoint
        if f1 > best_f1:
            best_f1, best_threshold = f1, best_t
            patience_cnt = 0
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "f1":            best_f1,
                "threshold":     best_threshold,
                "n_classes":     n_classes,
                "n_env":         n_env,
                "species_list":  species_list,
                "env_mean":      env_mean,
                "env_std":       env_std,
            }, os.path.join(args.save_dir, "best.pt"))
            print("  -> best.pt sauvegarde (F1={:.4f}, seuil={:.2f})".format(best_f1, best_threshold))
        else:
            patience_cnt += 1
            if args.patience > 0 and patience_cnt >= args.patience:
                print("  Early stopping.")
                break

    print("\nTermine. Meilleur F1: {:.4f} (seuil={:.2f})".format(best_f1, best_threshold))


# ============================================================================
# PREDICTION / SOUMISSION
# ============================================================================

def run_predict(args):
    import torch
    from torch.utils.data import DataLoader
    from model import BaselineModel, setup_device

    device = setup_device(args.gpu_id)

    ckpt         = torch.load(args.checkpoint, map_location=device, weights_only=False)
    species_list = ckpt["species_list"]
    env_mean     = ckpt["env_mean"]
    env_std      = ckpt["env_std"]
    n_classes    = ckpt["n_classes"]
    n_env        = ckpt["n_env"]
    threshold    = ckpt.get("threshold", 0.2)
    print("Checkpoint charge | F1={:.4f} | seuil={:.2f}".format(ckpt["f1"], threshold))

    model = BaselineModel(n_species=n_classes, n_env=n_env, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env_data, _ = load_all_env_features("test")

    # Lister les surveys test depuis les TIFF
    test_sids = []
    for root, _, files in os.walk(PATHS["pa_test"]):
        for f in files:
            if f.endswith(".tiff"):
                try:
                    test_sids.append(int(f.replace(".tiff", "")))
                except ValueError:
                    pass
    test_sids.sort()
    print("{} surveys test".format(len(test_sids)))

    PADataset = build_dataset_class()
    test_ds   = PADataset(test_sids, env_data, labels=None, species_list=species_list,
                          split="test", env_mean=env_mean, env_std=env_std, augment=False)
    loader    = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    use_amp   = (str(device) != "cpu")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    all_sids, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            env   = batch["env"].to(device,   non_blocking=True)
            if use_amp:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):
                    logits = model(image=image, env=env)
            else:
                logits = model(image=image, env=env)
            all_sids.extend(batch["survey_id"].tolist())
            all_probs.append(torch.sigmoid(logits).cpu())

    all_probs = torch.cat(all_probs)

    out_path = os.path.join(args.save_dir, "submission.csv")
    os.makedirs(args.save_dir, exist_ok=True)
    K = args.top_k
    with open(out_path, "w") as f:
        f.write("surveyId,predictions\n")
        for i, sid in enumerate(all_sids):
            row   = all_probs[i]
            above = (row >= threshold).nonzero(as_tuple=True)[0].tolist()
            if len(above) < K:
                above = row.topk(K).indices.tolist()
            sp_ids = sorted(species_list[j] for j in above)
            f.write("{},{}\n".format(sid, " ".join(str(s) for s in sp_ids)))

    print("Soumission : {}  ({} surveys)".format(out_path, len(all_sids)))


# ============================================================================
# EXPLORATION
# ============================================================================

def run_explore(args):
    print("\n" + "=" * 60)
    for name, path in PATHS.items():
        status = "OK  " if os.path.exists(path) else "MISS"
        print("  [{}] {} : {}".format(status, name, path))

    print("\nFeatures env (train):")
    env_data, names = load_all_env_features("train")
    print("  Exemple: {}".format(names[:6]))

    print("\nLabels :")
    labels, species, coords = load_labels_and_coords(LABELS_CSV)

    print("\nTest TIFF Sentinel-2:")
    sids  = list(env_data.keys())[:200]
    found = sum(1 for sid in sids if os.path.exists(get_tiff_path(sid)))
    print("  {}/{} premiers surveys ont un TIFF".format(found, len(sids)))
    for sid in sids:
        p = get_tiff_path(sid)
        if os.path.exists(p):
            img = load_sentinel_tiff(p)
            if img is not None:
                print("  survey {} : shape={}, min={:.3f}, max={:.3f}".format(
                    sid, img.shape, img.min(), img.max()))
            break

    print("\nDemo split spatial (block=2°, val_every=7):")
    valid = [sid for sid in labels if sid in env_data]
    spatial_split(valid, coords, block_deg=2.0, val_every=7)
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline GeoLifeCLEF 2025")

    parser.add_argument("--explore",    action="store_true")
    parser.add_argument("--predict",    action="store_true")
    parser.add_argument("--gpu-id",     type=int,   default=0)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--top-k",      type=int,   default=25,
                        help="Nombre minimum d'especes par survey (soumission)")
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early stopping (0 = desactive)")
    parser.add_argument("--block-deg",  type=float, default=2.0,
                        help="Taille des blocs spatiaux en degres (defaut: 2)")
    parser.add_argument("--val-every",  type=int,   default=7,
                        help="1 bloc sur val-every -> validation (~14%)")
    parser.add_argument("--labels",     type=str,   default=LABELS_CSV)
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/best.pt")
    parser.add_argument("--save-dir",   type=str,   default="checkpoints")

    args = parser.parse_args()

    if args.explore:
        run_explore(args)
    elif args.predict:
        run_predict(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
