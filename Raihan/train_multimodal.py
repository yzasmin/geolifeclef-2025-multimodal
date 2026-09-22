"""
train_multimodal.py - Pipeline multimodal GeoLifeCLEF MIASHS 2026
Auteur : groupe 1
Architecture : MLP(tabular) + GRU(Landsat TS) + MLP(Bioclim TS) → fusion → tête 5016 espèces
Validation : Leave-Region-Out (LRO) - scientifiquement rigoureux
Référence : Picek et al. 2024 (GeoPlant), Larcher et al. 2024 (Malpolon)
"""

import os, time, re, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_per_process_memory_fraction(0.25)  # 6 GB max

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42
BATCH_SIZE   = 256
EPOCHS       = 80
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 10          # early stopping
NUM_WORKERS  = 4

# Régions retirées pour la validation (leave-region-out)
# On choisit les régions les plus proches du test (STEPPIC absent du train)
# → on valide sur MEDITERRANEAN + ALPINE (diversité écologique max)
VAL_REGIONS  = {"MEDITERRANEAN", "ALPINE"}

# Chemins
DATA_DIR  = Path(os.environ.get("GLC_DATA_DIR", "data"))
ENV_DIR   = DATA_DIR / "EnvironmentalValues"
META_TRAIN= DATA_DIR / "GLC25_PA_metadata_train.csv"
META_TEST = DATA_DIR / "GLC25_PA_metadata_test.csv"
SAMPLE_SUB= DATA_DIR / "GLC25_SAMPLE_SUBMISSION.csv"

LANDSAT_TRAIN_DIR = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "SateliteTimeSeries-Landsat/cubes/PA-train")
LANDSAT_TEST_DIR  = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "SateliteTimeSeries-Landsat/cubes/PA-test")
BIO_TRAIN_DIR     = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "BioclimTimeSeries/cubes/PA-train")
BIO_TEST_DIR      = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "BioclimTimeSeries/cubes/PA-test")

SAVE_DIR = (Path(os.environ.get("GLC_OUTPUT_DIR", "artifacts")) / "checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────
#  1. CHARGEMENT LABELS
# ─────────────────────────────────────────────
def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df["speciesId"] = df["speciesId"].astype(int)
    species_list = sorted(df["speciesId"].unique())
    sp2idx = {sp: i for i, sp in enumerate(species_list)}
    labels = defaultdict(set)
    for row in df.itertuples():
        labels[row.surveyId].add(sp2idx[row.speciesId])
    return dict(labels), species_list, sp2idx

# ─────────────────────────────────────────────
#  2. CHARGEMENT FEATURES TABULAIRES
# ─────────────────────────────────────────────
def load_env(split="train"):
    suffix = "train" if split == "train" else "test"
    csvs = [
        ENV_DIR / f"ClimateAverage_1981-2010/GLC25-PA-{suffix}-bioclimatic.csv",
        ENV_DIR / f"SoilGrids/GLC25-PA-{suffix}-soilgrids.csv",
        ENV_DIR / f"Elevation/GLC25-PA-{suffix}-elevation.csv",
        ENV_DIR / f"LandCover/GLC25-PA-{suffix}-landcover.csv",
        ENV_DIR / f"HumanFootprint/GLC25-PA-{suffix}-human_footprint.csv",
    ]
    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        # Supprimer la colonne parasite 'Unnamed: 0' si présente
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = df.set_index("surveyId")
        dfs.append(df)
    merged = pd.concat(dfs, axis=1)
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged

# ─────────────────────────────────────────────
#  3. INDEX DES FICHIERS TEMPORELS
# ─────────────────────────────────────────────
def build_ts_index(directory, pattern):
    """Construit un dict surveyId → chemin fichier .pt"""
    idx = {}
    for f in Path(directory).glob("*.pt"):
        if f.name.startswith("._"):
            continue
        m = re.search(pattern, f.name)
        if m:
            idx[int(m.group(1))] = str(f)
    return idx

# ─────────────────────────────────────────────
#  4. IMPUTATION LANDSAT (NaN = nuages)
# ─────────────────────────────────────────────
def impute_landsat(t):
    """
    t: [6, 4, 21] = [bands, pixels, time]
    Stratégie : médiane temporelle par (bande, pixel)
    Référence : Ung et al. 2023 - cloud masking with temporal median
    """
    # Médiane sur la dim temporelle (dim=2), ignore NaN
    # Pour chaque (bande, pixel), remplace NaN par médiane sur le temps
    t_out = t.clone()
    for b in range(t.shape[0]):
        for p in range(t.shape[1]):
            ts = t[b, p, :]  # [21]
            mask = torch.isnan(ts)
            if mask.all():
                t_out[b, p, :] = 0.0
            elif mask.any():
                med = ts[~mask].median()
                t_out[b, p, mask] = med
    return t_out

# ─────────────────────────────────────────────
#  5. DATASET MULTIMODAL
# ─────────────────────────────────────────────
class MultiModalDataset(Dataset):
    def __init__(self, survey_ids, env_dict, landsat_idx, bio_idx,
                 labels=None, n_classes=5016):
        self.ids         = survey_ids
        self.env         = env_dict
        self.landsat_idx = landsat_idx
        self.bio_idx     = bio_idx
        self.labels      = labels
        self.n_classes   = n_classes

    def __len__(self):
        return len(self.ids)

    def _load_landsat(self, sid):
        """Charge et impute un cube Landsat [6,4,21]"""
        path = self.landsat_idx.get(sid)
        if path is None:
            return torch.zeros(6, 4, 21, dtype=torch.float32)
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
            t = impute_landsat(t)
            # Normalisation par bande (valeurs brutes 0-100)
            t = t / 100.0
            return t.float()
        except Exception:
            return torch.zeros(6, 4, 21, dtype=torch.float32)

    def _load_bioclim(self, sid):
        """Charge un cube Bioclim [4,19,12]"""
        path = self.bio_idx.get(sid)
        if path is None:
            return torch.zeros(4, 19, 12, dtype=torch.float32)
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
            # Normalisation simple - les valeurs sont dans [482, 16578]
            t = t / 10000.0
            return t.float()
        except Exception:
            return torch.zeros(4, 19, 12, dtype=torch.float32)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        # Features tabulaires
        x_env = self.env.get(sid)
        if x_env is None:
            x_env = np.zeros(self.env_dim, dtype=np.float32)
        x_env = torch.from_numpy(x_env.copy())
        x_env = torch.nan_to_num(x_env, nan=0.0, posinf=0.0, neginf=0.0)

        # Landsat [6, 4, 21]
        x_landsat = self._load_landsat(sid)

        # Bioclim TS [4, 19, 12]
        x_bio = self._load_bioclim(sid)

        if self.labels is not None:
            y = torch.zeros(self.n_classes, dtype=torch.float32)
            for sp_idx in self.labels.get(sid, set()):
                y[sp_idx] = 1.0
            return x_env, x_landsat, x_bio, y
        return x_env, x_landsat, x_bio, sid

# ─────────────────────────────────────────────
#  6. ARCHITECTURE MULTIMODALE
# ─────────────────────────────────────────────

class LandsatEncoder(nn.Module):
    """
    Encode une série temporelle Landsat [6, 4, 21]
    Approche : on flatten pixels → [6×4, 21] = [24, 21]
    Puis GRU bidirectionnel sur la dimension temporelle
    Inspiré de : Ung et al. 2023 (GeoLifeCLEF winner)
    """
    def __init__(self, out_dim=128):
        super().__init__()
        # [bands=6, pixels=4, time=21] → traiter comme [batch, seq=21, features=6*4=24]
        self.gru = nn.GRU(
            input_size=24,      # 6 bandes × 4 pixels
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),  # 128 × 2 directions
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, 6, 4, 21]
        B = x.shape[0]
        x = x.view(B, 24, 21)          # [B, 24, 21]
        x = x.permute(0, 2, 1)         # [B, 21, 24] = [batch, time, features]
        _, h = self.gru(x)              # h: [4, B, 128] (2 layers × 2 directions)
        # Concat les 2 directions de la dernière couche
        h = torch.cat([h[-2], h[-1]], dim=1)  # [B, 256]
        return self.proj(h)             # [B, out_dim]


class BioclimTSEncoder(nn.Module):
    """
    Encode la série temporelle mensuelle Bioclim [4, 19, 12]
    Approche : flatten pixels puis 1D-CNN sur les mois
    """
    def __init__(self, out_dim=64):
        super().__init__()
        # [4 pixels, 19 vars, 12 mois] → [4*19=76, 12]
        self.conv = nn.Sequential(
            nn.Conv1d(76, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, 4, 19, 12]
        B = x.shape[0]
        x = x.view(B, 76, 12)          # [B, 76, 12]
        x = self.conv(x)               # [B, 64, 12]
        x = self.pool(x).squeeze(-1)   # [B, 64]
        return self.proj(x)            # [B, out_dim]


class TabularEncoder(nn.Module):
    """
    MLP pour les features environnementales tabulaires + coords GPS sinusoïdales
    """
    def __init__(self, in_dim, out_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalModel(nn.Module):
    """
    Fusion tardive : tabular (256) + landsat (128) + bioclim (64) → 448 → 5016
    Référence architecture : Chen et al. 2024 (GeoLifeCLEF 2024 2nd place)
    """
    def __init__(self, tab_dim, n_classes, log_prior, dropout=0.3):
        super().__init__()
        self.tab_enc     = TabularEncoder(tab_dim, out_dim=256, dropout=dropout)
        self.landsat_enc = LandsatEncoder(out_dim=128)
        self.bio_enc     = BioclimTSEncoder(out_dim=64)

        fusion_dim = 256 + 128 + 64  # 448

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, n_classes),
        )

        # Initialisation du biais avec les log-odds de fréquence
        with torch.no_grad():
            self.fusion[-1].bias.copy_(torch.from_numpy(log_prior))

    def forward(self, x_tab, x_landsat, x_bio):
        e_tab     = self.tab_enc(x_tab)
        e_landsat = self.landsat_enc(x_landsat)
        e_bio     = self.bio_enc(x_bio)
        fused     = torch.cat([e_tab, e_landsat, e_bio], dim=1)
        return self.fusion(fused)


# ─────────────────────────────────────────────
#  7. PERTE ASYMETRIQUE
#  Référence : Ridnik et al. 2021 (ASL)
# ─────────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05):
        super().__init__()
        self.gp, self.gn, self.clip = gamma_pos, gamma_neg, clip

    def forward(self, logits, targets):
        p     = torch.sigmoid(logits)
        p_neg = (p + self.clip).clamp(max=1)
        lp    = (1 - p) ** self.gp * torch.log(p.clamp(1e-8))
        ln    = p_neg ** self.gn * torch.log((1 - p_neg).clamp(1e-8))
        return -(targets * lp + (1 - targets) * ln).mean()


# ─────────────────────────────────────────────
#  8. MÉTRIQUE F1 SAMPLE-AVERAGED
# ─────────────────────────────────────────────
def compute_f1_sample(preds_prob, targets, threshold):
    preds = (preds_prob > threshold).float()
    tp    = (preds * targets).sum(dim=1)
    fp    = (preds * (1 - targets)).sum(dim=1)
    fn    = ((1 - preds) * targets).sum(dim=1)
    f1    = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1.mean().item()


def find_best_threshold(probs, targets):
    """Cherche le meilleur seuil global sur [0.05, 0.60]"""
    best_f1, best_tau = 0.0, 0.15
    for tau in np.arange(0.05, 0.61, 0.01):
        f1 = compute_f1_sample(probs, targets, tau)
        if f1 > best_f1:
            best_f1, best_tau = f1, round(float(tau), 2)
    return best_tau, best_f1


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PIPELINE MULTIMODAL - GeoLifeCLEF MIASHS 2026")
    print("=" * 65)
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Labels ──
    print("\n[1/6] Chargement labels...")
    labels, species_list, sp2idx = load_labels(META_TRAIN)
    N_CLASSES = len(species_list)
    print(f"       {len(labels)} surveys | {N_CLASSES} espèces")

    # ── Features tabulaires ──
    print("\n[2/6] Chargement features tabulaires...")
    env_train_raw = load_env("train")
    env_test_raw  = load_env("test")

    # Imputation par médiane train
    col_medians   = env_train_raw.median()
    env_train_raw = env_train_raw.fillna(col_medians)
    env_test_raw  = env_test_raw.fillna(col_medians)

    # Normalisation Z-score (fit sur train uniquement)
    feat_mean = env_train_raw.mean()
    feat_std  = env_train_raw.std().replace(0, 1)
    env_train_norm = ((env_train_raw - feat_mean) / feat_std).astype(np.float32)
    env_test_norm  = ((env_test_raw  - feat_mean) / feat_std).fillna(0).astype(np.float32)

    # GPS sinusoïdaux - TRAIN
    meta_tr = pd.read_csv(META_TRAIN)[["surveyId","lon","lat"]].drop_duplicates("surveyId").set_index("surveyId")
    lon_tr, lat_tr = np.deg2rad(meta_tr["lon"].values), np.deg2rad(meta_tr["lat"].values)
    gps_tr = pd.DataFrame({
        "sin_lon": np.sin(lon_tr), "cos_lon": np.cos(lon_tr),
        "sin_lat": np.sin(lat_tr), "cos_lat": np.cos(lat_tr),
    }, index=meta_tr.index)

    # GPS sinusoïdaux - TEST (vraies coordonnées depuis metadata_test !)
    meta_te = pd.read_csv(META_TEST)[["surveyId","lon","lat"]].drop_duplicates("surveyId").set_index("surveyId")
    lon_te, lat_te = np.deg2rad(meta_te["lon"].values), np.deg2rad(meta_te["lat"].values)
    gps_te = pd.DataFrame({
        "sin_lon": np.sin(lon_te), "cos_lon": np.cos(lon_te),
        "sin_lat": np.sin(lat_te), "cos_lat": np.cos(lat_te),
    }, index=meta_te.index)

    env_train_final = pd.concat([env_train_norm, gps_tr.reindex(env_train_norm.index).fillna(0)], axis=1)
    env_test_final  = pd.concat([env_test_norm,  gps_te.reindex(env_test_norm.index).fillna(0)],  axis=1)

    N_TAB = env_train_final.shape[1]
    print(f"       Features tabulaires : {N_TAB} ({env_train_norm.shape[1]} env + 4 GPS)")

    # Dicts surveyId → array numpy
    env_train_dict = {sid: env_train_final.loc[sid].values for sid in env_train_final.index}
    env_test_dict  = {sid: env_test_final.loc[sid].values  for sid in env_test_final.index}

    # Corriger la dim pour le dataset
    MultiModalDataset.env_dim = N_TAB

    # ── Index séries temporelles ──
    print("\n[3/6] Indexation séries temporelles...")
    # Pattern Landsat train: GLC25-PA-train-landsat-time-series_XXXXXX_cube.pt
    # Pattern Landsat test : GLC25-PA-test-landsat_time_series_XXXXXX_cube.pt
    landsat_train_idx = build_ts_index(LANDSAT_TRAIN_DIR, r"_(\d+)_cube")
    landsat_test_idx  = build_ts_index(LANDSAT_TEST_DIR,  r"_(\d+)_cube")
    bio_train_idx     = build_ts_index(BIO_TRAIN_DIR,     r"_(\d+)_cube")
    bio_test_idx      = build_ts_index(BIO_TEST_DIR,      r"_(\d+)_cube")
    print(f"       Landsat  : {len(landsat_train_idx)} train | {len(landsat_test_idx)} test")
    print(f"       Bioclim  : {len(bio_train_idx)} train | {len(bio_test_idx)} test")

    # ── Split Leave-Region-Out ──
    print("\n[4/6] Split Leave-Region-Out...")
    meta_all = pd.read_csv(META_TRAIN)[["surveyId","region"]].drop_duplicates("surveyId")
    sid_to_region = dict(zip(meta_all["surveyId"], meta_all["region"]))

    all_sids = [s for s in labels if s in env_train_dict]
    val_sids = [s for s in all_sids if sid_to_region.get(s, "") in VAL_REGIONS]
    tr_sids  = [s for s in all_sids if sid_to_region.get(s, "") not in VAL_REGIONS]

    print(f"       Régions train : {sorted(set(sid_to_region[s] for s in tr_sids))}")
    print(f"       Régions val   : {sorted(VAL_REGIONS)}")
    print(f"       Surveys train : {len(tr_sids)} | val : {len(val_sids)}")

    # ── Prior de fréquence ──
    sp_counts = np.zeros(N_CLASSES, dtype=np.float32)
    for sid in tr_sids:
        for idx_sp in labels.get(sid, set()):
            sp_counts[idx_sp] += 1
    prevalence = (sp_counts + 1e-6) / (len(tr_sids) + 1e-6)
    log_prior  = np.log(prevalence / (1.0 - prevalence + 1e-6)).astype(np.float32)
    MEDIAN_LABELS = int(np.median([len(labels[s]) for s in tr_sids if s in labels]))
    print(f"       Médiane espèces/survey : {MEDIAN_LABELS}")

    # ── IDs test filtrés ──
    valid_test_ids = set(pd.read_csv(SAMPLE_SUB)["surveyId"].tolist())
    test_sids = sorted([s for s in env_test_dict if s in valid_test_ids])
    print(f"       Surveys test valides : {len(test_sids)}")

    # ── Datasets & DataLoaders ──
    def make_dataset(sids, env_dict, l_idx, b_idx, is_train):
        return MultiModalDataset(
            sids, env_dict, l_idx, b_idx,
            labels=labels if is_train else None,
            n_classes=N_CLASSES,
        )

    train_ds = make_dataset(tr_sids,   env_train_dict, landsat_train_idx, bio_train_idx, True)
    val_ds   = make_dataset(val_sids,  env_train_dict, landsat_train_idx, bio_train_idx, True)
    test_ds  = make_dataset(test_sids, env_test_dict,  landsat_test_idx,  bio_test_idx,  False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n       DataLoaders : train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # ── Modèle ──
    print("\n[5/6] Construction du modèle...")
    model = MultiModalModel(N_TAB, N_CLASSES, log_prior).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Paramètres : {n_params:,}")

    criterion = AsymmetricLoss(gamma_neg=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Entraînement ──
    print(f"\n[6/6] Entraînement ({EPOCHS} epochs max, patience={PATIENCE})...")
    print("-" * 65)

    best_f1, best_tau, patience_count = 0.0, 0.15, 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        # - Train -
        model.train()
        tr_loss, t0 = 0.0, time.time()

        for x_tab, x_l, x_b, y in tqdm(train_dl, desc=f"Ep {epoch:3d}/{EPOCHS}", leave=False):
            x_tab = x_tab.to(DEVICE)
            x_l   = x_l.to(DEVICE)
            x_b   = x_b.to(DEVICE)
            y     = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tab, x_l, x_b)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        scheduler.step()

        # - Validation -
        model.eval()
        all_p, all_y, val_loss = [], [], 0.0
        with torch.no_grad():
            for x_tab, x_l, x_b, y in val_dl:
                x_tab = x_tab.to(DEVICE)
                x_l   = x_l.to(DEVICE)
                x_b   = x_b.to(DEVICE)
                y     = y.to(DEVICE)
                logits = model(x_tab, x_l, x_b)
                val_loss += criterion(logits, y).item()
                all_p.append(torch.sigmoid(logits).cpu())
                all_y.append(y.cpu())

        all_p = torch.cat(all_p)
        all_y = torch.cat(all_y)
        tau, f1 = find_best_threshold(all_p, all_y)

        elapsed = time.time() - t0
        print(f"Ep {epoch:3d} | TrainL {tr_loss/len(train_dl):.4f} | "
              f"ValL {val_loss/len(val_dl):.4f} | "
              f"F1 {f1:.4f} (τ={tau:.2f}) | {elapsed:.0f}s")

        # - Sauvegarde meilleur modèle -
        if f1 > best_f1:
            best_f1, best_tau, best_epoch = f1, tau, epoch
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "f1": best_f1,
                "tau": best_tau,
                "species_list": species_list,
                "feat_mean": feat_mean.values,
                "feat_std": feat_std.values,
                "col_medians": col_medians.values,
                "n_tab": N_TAB,
                "n_classes": N_CLASSES,
                "median_labels": MEDIAN_LABELS,
                "val_regions": list(VAL_REGIONS),
            }, SAVE_DIR / "best_multimodal.pt")
            print(f"  ✓ Sauvegardé (F1={best_f1:.4f}, τ={best_tau:.2f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n→ Early stopping à l'epoch {epoch} (patience={PATIENCE})")
                print(f"  Meilleur modèle : epoch {best_epoch}, F1={best_f1:.4f}")
                break

    print(f"\n{'='*65}")
    print(f"Meilleur F1 val (LRO) : {best_f1:.4f}  τ={best_tau:.2f}  epoch={best_epoch}")
    print(f"{'='*65}")

    # ── Génération soumission ──
    print("\nGénération de la soumission...")
    ckpt = torch.load(SAVE_DIR / "best_multimodal.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tau           = ckpt["tau"]
    species_list  = ckpt["species_list"]
    median_labels = ckpt["median_labels"]

    rows = []
    with torch.no_grad():
        for x_tab, x_l, x_b, sids in tqdm(test_dl, desc="Prédiction test"):
            x_tab = x_tab.to(DEVICE)
            x_l   = x_l.to(DEVICE)
            x_b   = x_b.to(DEVICE)
            probs = torch.sigmoid(model(x_tab, x_l, x_b)).cpu()

            for i, sid in enumerate(sids.tolist()):
                idx = (probs[i] > tau).nonzero(as_tuple=True)[0].tolist()
                if not idx:  # fallback : top-k
                    idx = probs[i].topk(median_labels).indices.tolist()
                sp_ids = sorted([species_list[j] for j in idx])
                rows.append({"surveyId": sid, "predictions": " ".join(map(str, sp_ids))})

    sub = pd.DataFrame(rows)

    # Vérification finale
    assert len(sub) == 14784, f"ERREUR : {len(sub)} lignes au lieu de 14784 !"
    assert set(sub["surveyId"].tolist()) == valid_test_ids, "ERREUR : IDs ne correspondent pas !"

    out_path = SAVE_DIR / "submission_multimodal.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSoumission sauvegardée : {out_path}")
    print(f"Surveys : {len(sub)} | Espèces moyennes : {sub['predictions'].apply(lambda x: len(x.split())).mean():.1f}")
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()
