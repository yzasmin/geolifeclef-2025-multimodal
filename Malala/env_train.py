# -*- coding: utf-8 -*-
"""
Entrainement de la branche Environnement - EnvironmentalMLP
============================================================

Donnees utilisees :
  features X : bioclim (19) + soilgrids (9) = 28 scalaires par survey
  labels   y : vecteur multi-hot (n_species,) PA

Split : spatial (blocs lat/lon) pour eviter la fuite d'information geographique.

Usage :
    python3 env_train.py --explore
    python3 env_train.py --epochs 50 --batch-size 512 --gpu-id 0
    python3 env_train.py --predict --checkpoint checkpoints/env_best.pt
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# CHEMINS
# ============================================================================

BASE_DIR   = "/data/challenge2026MIASHS"
LABELS_CSV = os.path.expanduser("~/GLC25_PA_metadata_train.csv")

PATHS = {
    "bioclim_train":   os.path.join(BASE_DIR, "EnvironmentalValues", "ClimateAverage_1981-2010", "GLC25-PA-train-bioclimatic.csv"),
    "bioclim_test":    os.path.join(BASE_DIR, "EnvironmentalValues", "ClimateAverage_1981-2010", "GLC25-PA-test-bioclimatic.csv"),
    "soilgrids_train": os.path.join(BASE_DIR, "EnvironmentalValues", "SoilGrids", "GLC25-PA-train-soilgrids.csv"),
    "soilgrids_test":  os.path.join(BASE_DIR, "EnvironmentalValues", "SoilGrids", "GLC25-PA-test-soilgrids.csv"),
}


# ============================================================================
# CHARGEMENT
# ============================================================================

def load_csv_as_dict(path):
    # type: (str) -> Tuple[Dict[int, np.ndarray], List[str]]
    """Charge un CSV env : {surveyId → np.array}, noms des colonnes."""
    data = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
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


def load_env_features(split="train"):
    # type: (str) -> Tuple[Dict[int, np.ndarray], List[str]]
    """
    Charge et concatene bioclim + soilgrids pour un split donne.

    Retourne :
      env_data : {surveyId → np.array(28,)}   (19 bioclim + 9 soilgrids)
      names    : liste des 28 noms de features
    """
    sources = [
        ("bioclim_" + split,   "bioclim"),
        ("soilgrids_" + split, "soilgrids"),
    ]

    all_parts  = {}   # {sid: [array_bioclim, array_soilgrids]}
    all_names  = []

    for key, label in sources:
        path = PATHS[key]
        if not os.path.exists(path):
            print("  WARN: {} introuvable".format(path))
            continue
        data, names = load_csv_as_dict(path)
        print("  {} : {} surveys, {} features".format(label, len(data), len(names)))
        all_names.extend(names)
        for sid, vals in data.items():
            all_parts.setdefault(sid, []).append(vals)

    n_feats  = len(all_names)
    env_data = {}
    for sid, parts in all_parts.items():
        concat = np.concatenate(parts)
        if len(concat) == n_feats:
            env_data[sid] = concat

    print("  Total : {} surveys, {} features".format(len(env_data), n_feats))
    return env_data, all_names


def load_labels_and_coords(labels_csv):
    # type: (str) -> Tuple[Dict[int, set], List[int], Dict[int, Tuple[float, float]]]
    """
    Charge le CSV metadata PA.
    Retourne : labels {surveyId: {spId}}, species triees, coords {surveyId: (lon, lat)}.
    """
    if not os.path.exists(labels_csv):
        print("ERREUR : {} introuvable".format(labels_csv))
        sys.exit(1)

    labels  = defaultdict(set)
    coords  = {}
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
                labels[sid].add(sp)
                species.add(sp)
                coords[sid] = (float(row[lon_col]), float(row[lat_col]))
            except (ValueError, IndexError):
                continue

    counts = [len(v) for v in labels.values()]
    print("  {} surveys | {} especes | moy {:.1f} esp/survey".format(
        len(labels), len(species), sum(counts) / max(len(counts), 1)))

    return dict(labels), sorted(species), coords


def compute_norm_stats(env_data):
    # type: (Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]
    """Z-score : mean et std calcules sur l'ensemble d'entrainement uniquement."""
    matrix = np.stack(list(env_data.values()))
    mean   = matrix.mean(axis=0)
    std    = matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def compute_pos_weight(labels, species_list):
    # type: (Dict[int, set], List[int]) -> torch.Tensor
    """
    pos_weight[k] = n_negatifs / n_positifs pour chaque espece, plafonne a 100.
    Compense le fort desequilibre : la plupart des especes sont absentes
    dans la majorite des surveys.
    """
    n      = len(labels)
    sp2idx = {sp: i for i, sp in enumerate(species_list)}
    pos    = torch.zeros(len(species_list))
    for sp_set in labels.values():
        for sp in sp_set:
            if sp in sp2idx:
                pos[sp2idx[sp]] += 1
    pos_weight = ((n - pos) / (pos + 1e-8)).clamp(max=100.0)
    print("  pos_weight | min={:.1f}  max={:.1f}  mean={:.1f}".format(
        pos_weight.min().item(), pos_weight.max().item(), pos_weight.mean().item()))
    return pos_weight


# ============================================================================
# SPLIT SPATIAL
# ============================================================================

def spatial_split(survey_ids, coords, block_deg=2.0, val_every=7):
    # type: (List[int], Dict[int, Tuple[float, float]], float, int) -> Tuple[List[int], List[int]]
    """
    Divise les surveys selon un decoupage geographique en blocs de `block_deg` degres.
    1 bloc sur `val_every` va en validation.

    Avantage vs split aleatoire : evite la fuite d'information due a
    l'autocorrelation spatiale (surveys voisins = especes similaires).
    """
    sid_to_cell = {
        sid: (int(coords[sid][0] // block_deg), int(coords[sid][1] // block_deg))
        if sid in coords else (0, 0)
        for sid in survey_ids
    }

    cells     = sorted(set(sid_to_cell.values()))
    val_cells = {cells[i] for i in range(0, len(cells), val_every)}

    train = [sid for sid in survey_ids if sid_to_cell[sid] not in val_cells]
    val   = [sid for sid in survey_ids if sid_to_cell[sid] in val_cells]

    print("  Blocs total={} | train={} | val={} ({:.1f}%)".format(
        len(cells), len(cells) - len(val_cells), len(val_cells),
        100.0 * len(val) / max(1, len(survey_ids))))
    print("  Surveys -> train: {}  val: {}".format(len(train), len(val)))
    return train, val


# ============================================================================
# DATASET
# ============================================================================

class EnvironmentalDataset(Dataset):
    """
    Dataset PA focalise sur les features environnementales.

    Chaque item retourne :
      X : tenseur (n_features,) normalise en Z-score
      y : tenseur (n_species,) multi-hot binaire

    Args:
        survey_ids   : liste des surveyId a inclure
        env_data     : {surveyId → np.array brut}
        labels       : {surveyId → set(spId)}  - None pour le test
        species_list : liste ordonnee des especes (definit l'ordre de y)
        env_mean     : mean pour normalisation Z-score (calculee sur le train)
        env_std      : std  pour normalisation Z-score
    """

    def __init__(
        self,
        survey_ids,         # type: List[int]
        env_data,           # type: Dict[int, np.ndarray]
        labels=None,        # type: Optional[Dict[int, set]]
        species_list=None,  # type: Optional[List[int]]
        env_mean=None,      # type: Optional[np.ndarray]
        env_std=None,       # type: Optional[np.ndarray]
    ):
        self.survey_ids  = survey_ids
        self.env_data    = env_data
        self.labels      = labels
        self.env_mean    = env_mean
        self.env_std     = env_std
        self.n_species   = len(species_list) if species_list else 0
        self.sp2idx      = {sp: i for i, sp in enumerate(species_list)} if species_list else {}
        self.n_features  = len(env_mean) if env_mean is not None else 28

    def __len__(self):
        return len(self.survey_ids)

    def __getitem__(self, idx):
        sid = self.survey_ids[idx]

        # -- Features X --
        if sid in self.env_data:
            x = self.env_data[sid].copy()
            if self.env_mean is not None:
                x = (x - self.env_mean) / self.env_std
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x = np.zeros(self.n_features, dtype=np.float32)

        X = torch.from_numpy(x.astype(np.float32))

        # -- Labels y --
        if self.labels is not None and sid in self.labels:
            y_arr = np.zeros(self.n_species, dtype=np.float32)
            for sp in self.labels[sid]:
                if sp in self.sp2idx:
                    y_arr[self.sp2idx[sp]] = 1.0
            y = torch.from_numpy(y_arr)
        else:
            y = torch.zeros(self.n_species, dtype=torch.float32)

        return X, y, sid


# ============================================================================
# METRIQUES
# ============================================================================

def compute_f1(preds_bin, labels):
    # type: (torch.Tensor, torch.Tensor) -> float
    """F1 sample-averaged - metrique officielle du challenge."""
    tp = (preds_bin * labels).sum(dim=-1)
    fp = (preds_bin * (1 - labels)).sum(dim=-1)
    fn = ((1 - preds_bin) * labels).sum(dim=-1)
    return (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()


def find_best_threshold(probs, labels):
    # type: (torch.Tensor, torch.Tensor) -> Tuple[float, float]
    """Cherche le seuil optimal sur 10 valeurs entre 0.05 et 0.50."""
    best_t, best_f1 = 0.2, 0.0
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        f1 = compute_f1((probs > t).float(), labels)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ============================================================================
# ENTRAINEMENT
# ============================================================================

def run_train(args):
    from model import setup_device   # GPU helper commun
    from env_model import EnvironmentalMLP

    device = setup_device(args.gpu_id)

    # 1. Donnees
    print("\n[1/4] Labels + coordonnees...")
    labels, species_list, coords = load_labels_and_coords(LABELS_CSV)
    n_species = len(species_list)

    print("\n[2/4] Features environnementales (bioclim + soilgrids)...")
    env_data, feat_names = load_env_features("train")
    n_features = len(feat_names)
    print("  Features : {}".format(feat_names))

    # Normalisation Z-score (uniquement sur les surveys d'entrainement)
    valid_sids = [sid for sid in labels if sid in env_data]
    env_subset = {sid: env_data[sid] for sid in valid_sids}
    env_mean, env_std = compute_norm_stats(env_subset)

    # 2. Split spatial
    print("\n[3/4] Split spatial (blocs {}°)...".format(args.block_deg))
    train_sids, val_sids = spatial_split(
        valid_sids, coords,
        block_deg=args.block_deg,
        val_every=args.val_every,
    )

    # 3. Datasets + DataLoaders
    train_ds = EnvironmentalDataset(train_sids, env_data, labels=labels,
                                    species_list=species_list,
                                    env_mean=env_mean, env_std=env_std)
    val_ds   = EnvironmentalDataset(val_sids, env_data, labels=labels,
                                    species_list=species_list,
                                    env_mean=env_mean, env_std=env_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    print("  Train: {}  Val: {}".format(len(train_ds), len(val_ds)))

    # 4. Modele
    print("\n[4/4] Modele...")
    model  = EnvironmentalMLP(n_features=n_features, n_species=n_species).to(device)
    n_pars = sum(p.numel() for p in model.parameters())
    print("  EnvironmentalMLP | {:,} parametres | features={}".format(n_pars, n_features))

    # pos_weight : compense le desequilibre especes rares
    print("  Calcul pos_weight...")
    pos_weight = compute_pos_weight(
        {sid: labels[sid] for sid in train_sids}, species_list
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1        = 0.0
    best_threshold = 0.2
    patience_cnt   = 0

    print("\n" + "=" * 65)
    print("Entrainement | {} epochs | batch {} | lr {}".format(
        args.epochs, args.batch_size, args.lr))
    print("=" * 65 + "\n")

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        t_loss, n_b = 0.0, 0
        t0 = time.time()

        for X, y, _ in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item()
            n_b    += 1

        scheduler.step()

        # ── Validation ──
        model.eval()
        v_loss, n_vb          = 0.0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits  = model(X)
                loss    = criterion(logits, y)
                v_loss += loss.item()
                n_vb   += 1

                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(y.cpu())

        probs_cat  = torch.cat(all_probs)
        labels_cat = torch.cat(all_labels)
        best_t, f1 = find_best_threshold(probs_cat, labels_cat)

        print("Epoch {:3d}/{} | loss {:.4f} | val_loss {:.4f} | F1@{:.2f}={:.4f} | {:.0f}s".format(
            epoch + 1, args.epochs,
            t_loss / max(n_b, 1),
            v_loss / max(n_vb, 1),
            best_t, f1,
            time.time() - t0))

        # Sauvegarde du meilleur checkpoint
        if f1 > best_f1:
            best_f1, best_threshold = f1, best_t
            patience_cnt = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "f1":           best_f1,
                "threshold":    best_threshold,
                "n_features":   n_features,
                "n_species":    n_species,
                "feat_names":   feat_names,
                "species_list": species_list,
                "env_mean":     env_mean,
                "env_std":      env_std,
            }, os.path.join(args.save_dir, "env_best.pt"))
            print("  -> env_best.pt (F1={:.4f}, seuil={:.2f})".format(best_f1, best_threshold))
        else:
            patience_cnt += 1
            if args.patience > 0 and patience_cnt >= args.patience:
                print("  Early stopping (patience={})".format(args.patience))
                break

    print("\nTermine | Meilleur F1: {:.4f} (seuil={:.2f})".format(best_f1, best_threshold))


# ============================================================================
# PREDICTION
# ============================================================================

def run_predict(args):
    from model import setup_device
    from env_model import EnvironmentalMLP

    device = setup_device(args.gpu_id)

    ckpt         = torch.load(args.checkpoint, map_location=device, weights_only=False)
    species_list = ckpt["species_list"]
    env_mean     = ckpt["env_mean"]
    env_std      = ckpt["env_std"]
    n_features   = ckpt["n_features"]
    n_species    = ckpt["n_species"]
    threshold    = ckpt.get("threshold", 0.2)
    print("Checkpoint | F1={:.4f} | seuil={:.2f} | features={}".format(
        ckpt["f1"], threshold, n_features))

    model = EnvironmentalMLP(n_features=n_features, n_species=n_species)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    env_data, _ = load_env_features("test")
    test_sids   = sorted(env_data.keys())
    print("{} surveys test".format(len(test_sids)))

    test_ds  = EnvironmentalDataset(test_sids, env_data, labels=None,
                                    species_list=species_list,
                                    env_mean=env_mean, env_std=env_std)
    loader   = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    all_sids, all_probs = [], []
    with torch.no_grad():
        for X, _, sids in loader:
            X      = X.to(device, non_blocking=True)
            logits = model(X)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_sids.extend(sids.tolist() if hasattr(sids, "tolist") else list(sids))

    all_probs = torch.cat(all_probs)

    out_path = os.path.join(args.save_dir, "env_submission.csv")
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
    print("EXPLORATION - Branche Environnement")
    print("=" * 60)

    for key, path in PATHS.items():
        status = "OK  " if os.path.exists(path) else "MISS"
        print("  [{}] {}".format(status, path))

    print("\nChargement bioclim + soilgrids (train):")
    env_data, names = load_env_features("train")
    print("  Features ({}) : {}".format(len(names), names))

    arr = np.stack(list(env_data.values()))
    print("  Stats (sur {} surveys) :".format(len(env_data)))
    print("    min  = {}".format(arr.min(axis=0).round(3)))
    print("    max  = {}".format(arr.max(axis=0).round(3)))
    print("    mean = {}".format(arr.mean(axis=0).round(3)))

    print("\nLabels:")
    labels, species, coords = load_labels_and_coords(LABELS_CSV)
    valid = [sid for sid in labels if sid in env_data]
    print("  Surveys valides (labels + env) : {}".format(len(valid)))

    print("\nDemo split spatial:")
    spatial_split(valid, coords, block_deg=args.block_deg, val_every=args.val_every)

    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Branche Environnement - GeoLifeCLEF 2025")

    parser.add_argument("--explore",    action="store_true")
    parser.add_argument("--predict",    action="store_true")
    parser.add_argument("--gpu-id",     type=int,   default=0)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=512,
                        help="Batch large possible : donnees tabulaires, pas d'images")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--top-k",      type=int,   default=25)
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Early stopping (0 = desactive)")
    parser.add_argument("--block-deg",  type=float, default=2.0)
    parser.add_argument("--val-every",  type=int,   default=7)
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/env_best.pt")
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
