"""
train_final.py - Pipeline multimodal GeoLifeCLEF MIASHS 2026
Version finale - code unique autonome

Fondements scientifiques :
  - Picek et al. 2024 (GeoPlant dataset)
  - Chen et al. 2024 (GeoLifeCLEF 2024 2nd place - Transformer multimodal)
  - Ung et al. 2023 (GeoLifeCLEF 2023 winner - seuil adaptatif)
  - Ridnik et al. 2021 (AsymmetricLoss)
  - Roberts et al. 2017 (Leave-Region-Out validation)
  - Vaswani et al. 2017 (Positional encoding)

Corrections vs versions précédentes :
  1. log1p sur variables très asymétriques (skew>3)
  2. Suppression des features redondantes (r>0.95)
  3. z-score par bande Landsat (valeurs 0-254, pas 0-100)
  4. z-score par variable Bioclim (valeurs ~3800, pas ~1)
  5. Transformer avec positional encoding (ordre temporel)
  6. Attention pooling (dates informatives pondérées)
  7. Gating multi-modal appris
  8. Seuil adaptatif par fréquence d'espèce
  9. Initialisation biais par log-odds de prévalence
  10. Imputation NaN Landsat par médiane temporelle
"""

import os, sys, time, re, math, random, warnings
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

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_per_process_memory_fraction(0.25)   # 6 GB / 24 GB

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42
BATCH_SIZE   = 256          # plus grand → gradient plus stable
EPOCHS       = 100
LR           = 3e-4         # réduit : 5e-4 → 3e-4
WEIGHT_DECAY = 1e-3         # augmenté : 1e-4 → 1e-3 (régularisation L2)
PATIENCE     = 15           # augmenté pour laisser plus de temps
NUM_WORKERS  = 2            # réduit pour respecter limite CPU

# Validation Leave-Region-Out
# ALPINE (×8 dans test) + MEDITERRANEAN (région la plus diverse)
VAL_REGIONS = {"MEDITERRANEAN", "ALPINE"}

# Features redondantes à supprimer (r>0.95 entre paires)
DROP_FEATURES = {"Bio13", "Bio14", "Bio1", "LandCover-4", "LandCover-6"}

# Variables à transformer en log1p (|skew| > 3)
LOG_FEATURES = {
    "Bio12", "Bio16", "Bio19", "Elevation",
    "HumanFootprint-cemetery", "HumanFootprint-reservoir",
    "HumanFootprint-greenhouse", "HumanFootprint-farmland",
    "HumanFootprint-building-copernicus", "HumanFootprint-harbour",
    "HumanFootprint-building-residential", "HumanFootprint-building-commercial",
    "HumanFootprint-grass", "HumanFootprint-quarry", "HumanFootprint-road",
    "HumanFootprint-salt", "HumanFootprint-building-industrial",
    "HumanFootprint-vineyard", "HumanFootprint-military",
    "HumanFootprint-construction-site", "HumanFootprint-orchard",
    "HumanFootprint-farmyard", "HumanFootprint-dump-site",
    "HumanFootprint-railway",
}

# Chemins
DATA_DIR  = Path(os.environ.get("GLC_DATA_DIR", "data"))
ENV_DIR   = DATA_DIR / "EnvironmentalValues"
META_TRAIN= DATA_DIR / "GLC25_PA_metadata_train.csv"
META_TEST = DATA_DIR / "GLC25_PA_metadata_test.csv"
SAMPLE_SUB= DATA_DIR / "GLC25_SAMPLE_SUBMISSION.csv"

LANDSAT_TRAIN = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "SateliteTimeSeries-Landsat/cubes/PA-train")
LANDSAT_TEST  = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "SateliteTimeSeries-Landsat/cubes/PA-test")
BIO_TRAIN     = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "BioclimTimeSeries/cubes/PA-train")
BIO_TEST      = (Path(os.environ.get("GLC_DATA_DIR", "data")) / "BioclimTimeSeries/cubes/PA-test")

SAVE_DIR = (Path(os.environ.get("GLC_OUTPUT_DIR", "artifacts")) / "checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ══════════════════════════════════════════════════════════════
#  1. LABELS
# ══════════════════════════════════════════════════════════════
def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df["speciesId"] = df["speciesId"].astype(int)
    species_list = sorted(df["speciesId"].unique())
    sp2idx = {sp: i for i, sp in enumerate(species_list)}
    labels = defaultdict(set)
    for row in df.itertuples():
        labels[row.surveyId].add(sp2idx[row.speciesId])
    return dict(labels), species_list, sp2idx

# ══════════════════════════════════════════════════════════════
#  2. FEATURES TABULAIRES
#  Corrections : log1p sur variables asymétriques,
#                suppression features redondantes
# ══════════════════════════════════════════════════════════════
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
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = df.set_index("surveyId")
        dfs.append(df)
    merged = pd.concat(dfs, axis=1)
    merged = merged[~merged.index.duplicated(keep="first")]

    # Supprimer features redondantes (r>0.95)
    cols_to_drop = [c for c in DROP_FEATURES if c in merged.columns]
    merged = merged.drop(columns=cols_to_drop)

    # log1p sur variables très asymétriques
    # Clamp les négatifs à 0 avant log (certaines valeurs peuvent être légèrement < 0)
    for col in merged.columns:
        if col in LOG_FEATURES:
            merged[col] = np.log1p(merged[col].clip(lower=0))

    return merged


def encode_gps(meta_df):
    """GPS sinusoïdal - encodage circulaire correct"""
    lo = np.deg2rad(meta_df["lon"].values)
    la = np.deg2rad(meta_df["lat"].values)
    return pd.DataFrame({
        "sin_lon": np.sin(lo), "cos_lon": np.cos(lo),
        "sin_lat": np.sin(la), "cos_lat": np.cos(la),
    }, index=meta_df.index, dtype=np.float32)

# ══════════════════════════════════════════════════════════════
#  3. INDEX FICHIERS TEMPORELS
# ══════════════════════════════════════════════════════════════
def build_index(directory, pattern=r"_(\d+)_cube"):
    idx = {}
    for f in Path(directory).glob("*.pt"):
        if f.name.startswith("._"):
            continue
        m = re.search(pattern, f.name)
        if m:
            idx[int(m.group(1))] = str(f)
    return idx

# ══════════════════════════════════════════════════════════════
#  4. STATISTIQUES DE NORMALISATION
#  Correction principale : z-score par bande/variable
#  Référence : Malpolon (Larcher et al. 2024)
# ══════════════════════════════════════════════════════════════
def compute_stats(file_idx, dim_axis, n_dims, n_sample=500, desc=""):
    """
    Calcule mean et std pour normalisation z-score.
    dim_axis : axe sur lequel calculer les stats (0=bandes, 1=variables)
    """
    files = random.sample(list(file_idx.values()), min(n_sample, len(file_idx)))
    sums  = np.zeros(n_dims, dtype=np.float64)
    sums2 = np.zeros(n_dims, dtype=np.float64)
    cnts  = np.zeros(n_dims, dtype=np.int64)

    for f in tqdm(files, desc=f"  Stats {desc}", leave=False):
        try:
            t = torch.load(f, map_location="cpu", weights_only=True).numpy()
            for d in range(n_dims):
                if dim_axis == 0:
                    vals = t[d].flatten()
                else:
                    vals = t[:, d, :].flatten()
                vals = vals[~np.isnan(vals)]
                sums[d]  += vals.sum()
                sums2[d] += (vals ** 2).sum()
                cnts[d]  += len(vals)
        except:
            pass

    mean = sums / np.maximum(cnts, 1)
    std  = np.sqrt(np.maximum(sums2 / np.maximum(cnts, 1) - mean ** 2, 1e-6))
    return (torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std,  dtype=torch.float32))

# ══════════════════════════════════════════════════════════════
#  5. IMPUTATION LANDSAT (NaN = nuages)
#  Référence : Ung et al. 2023
# ══════════════════════════════════════════════════════════════
def impute_landsat(t):
    """Remplace NaN par médiane temporelle par (bande, pixel)"""
    t_out = t.clone()
    for b in range(t.shape[0]):
        for p in range(t.shape[1]):
            ts = t[b, p]; mask = torch.isnan(ts)
            if mask.all():    t_out[b, p] = 0.0
            elif mask.any():  t_out[b, p, mask] = ts[~mask].median()
    return t_out

# ══════════════════════════════════════════════════════════════
#  6. DATASET
# ══════════════════════════════════════════════════════════════
class PlantDataset(Dataset):
    def __init__(
        self, survey_ids,
        env_dict, aux_dict,
        l_idx, b_idx,
        l_mean, l_std,
        b_mean, b_std,
        labels=None, n_classes=5016,
    ):
        self.ids       = survey_ids
        self.env       = env_dict
        self.aux       = aux_dict
        self.l_idx     = l_idx
        self.b_idx     = b_idx
        # Shapes pour broadcast : [6,1,1] et [1,19,1]
        self.l_mean    = l_mean.view(6, 1, 1)
        self.l_std     = l_std.view(6, 1, 1)
        self.b_mean    = b_mean.view(1, 19, 1)
        self.b_std     = b_std.view(1, 19, 1)
        self.labels    = labels
        self.n_classes = n_classes

    def __len__(self): return len(self.ids)

    def _landsat(self, sid):
        """
        Charge Landsat [6,4,21], impute NaN, normalise z-score par bande.
        Reshape en [21, 24] pour le Transformer (seq_len, features).
        """
        path = self.l_idx.get(sid)
        if not path:
            return torch.zeros(21, 24, dtype=torch.float32)
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
            t = impute_landsat(t)                   # [6, 4, 21]
            t = (t - self.l_mean) / self.l_std      # z-score par bande
            # [6, 4, 21] → permute → [21, 6, 4] → reshape → [21, 24]
            return t.permute(2, 0, 1).reshape(21, 24).float()
        except:
            return torch.zeros(21, 24, dtype=torch.float32)

    def _bioclim(self, sid):
        """
        Charge Bioclim [4,19,12], normalise z-score par variable.
        Reshape en [12, 76] pour le Transformer (seq_len, features).
        """
        path = self.b_idx.get(sid)
        if not path:
            return torch.zeros(12, 76, dtype=torch.float32)
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
            t = (t - self.b_mean) / self.b_std      # z-score par variable
            # [4, 19, 12] → permute → [12, 4, 19] → reshape → [12, 76]
            return t.permute(2, 0, 1).reshape(12, 76).float()
        except:
            return torch.zeros(12, 76, dtype=torch.float32)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        x_env = torch.from_numpy(
            self.env.get(sid, np.zeros(PlantDataset.env_dim, dtype=np.float32)).copy()
        ).float()
        x_env = torch.nan_to_num(x_env, nan=0., posinf=0., neginf=0.)

        x_aux = torch.from_numpy(
            self.aux.get(sid, np.zeros(4, dtype=np.float32)).copy()
        ).float()

        x_l = self._landsat(sid)
        x_b = self._bioclim(sid)

        if self.labels is not None:
            y = torch.zeros(self.n_classes, dtype=torch.float32)
            for sp in self.labels.get(sid, set()):
                y[sp] = 1.0
            return x_env, x_aux, x_l, x_b, y
        return x_env, x_aux, x_l, x_b, sid

# ══════════════════════════════════════════════════════════════
#  7. ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    """
    Positional Encoding sinusoïdal.
    AJOUT vs modèle collègue : le Transformer sait que janvier ≠ juillet.
    Référence : Vaswani et al. 2017
    """
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class AttentionPool(nn.Module):
    """
    Attention pooling : pondère les timesteps selon leur informativité.
    AJOUT vs modèle collègue : au lieu de mean(dim=1) naïf.
    Ex: pic de végétation juillet > décembre pour annuelles.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, seq, d]
        w = F.softmax(self.w(x), dim=1)   # [B, seq, 1]
        return (w * x).sum(dim=1)          # [B, d]


class TSEncoder(nn.Module):
    """
    Encodeur Transformer pour séries temporelles.
    v2 anti-overfitting :
      - n_layers réduit (3→2 Landsat, 2→1 Bioclim)
      - dropout augmenté (0.1→0.3)
      - dim_feedforward réduit (4x→2x)
      - d_model réduit (128→64)
    """
    def __init__(self, in_features, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.3, max_len=128):
        super().__init__()
        self.proj   = nn.Linear(in_features, d_model)
        self.pe     = SinusoidalPE(d_model, max_len, dropout)
        enc_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,   # réduit : 4x → 2x
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.tf   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)

    def forward(self, x):
        x = self.proj(x)
        x = self.pe(x)
        x = self.tf(x)
        x = self.norm(x)
        return self.pool(x)


class MLPEnc(nn.Module):
    """MLP pour features tabulaires - dropout augmenté anti-overfitting"""
    def __init__(self, in_dim, out_dim, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class ModalGating(nn.Module):
    """
    Gating multi-modal : apprend le poids de chaque modalité.
    AJOUT vs modèle collègue : fusion pondérée au lieu de concat simple.
    Référence : Chen et al. 2024
    """
    def __init__(self, n_mod, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(n_mod * d, n_mod), nn.Sigmoid())

    def forward(self, embs):
        # embs : liste de tenseurs [B, d]
        cat     = torch.cat(embs, dim=1)                # [B, n_mod*d]
        weights = self.gate(cat).unsqueeze(-1)           # [B, n_mod, 1]
        stacked = torch.stack(embs, dim=1)              # [B, n_mod, d]
        gated   = (stacked * weights).reshape(cat.shape[0], -1)
        return gated                                     # [B, n_mod*d]


class FusionHead(nn.Module):
    """Tête de classification avec residual"""
    def __init__(self, in_dim, n_classes, hidden=512, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.res        = nn.Linear(in_dim, hidden)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.classifier(self.block(x) + self.res(x))


class MultimodalModel(nn.Module):
    """
    Modèle final.
    4 branches : env_MLP + aux_MLP + landsat_Transformer + bioclim_Transformer
    Gating multi-modal + residual fusion head
    """
    def __init__(self, env_dim, aux_dim, n_classes, log_prior, d=64, dropout=0.3):
        super().__init__()
        self.env_enc     = MLPEnc(env_dim, d, (256, 128), dropout)
        self.aux_enc     = MLPEnc(aux_dim, d, (64,  64),  dropout)
        # Landsat : 2 couches, Bioclim : 1 couche - anti-overfitting
        self.landsat_enc = TSEncoder(24, d, n_heads=4, n_layers=2, dropout=dropout, max_len=21)
        self.bioclim_enc = TSEncoder(76, d, n_heads=4, n_layers=1, dropout=dropout, max_len=12)
        self.gating      = ModalGating(4, d)
        self.head        = FusionHead(4 * d, n_classes, hidden=256, dropout=dropout)

        # Initialisation biais par log-odds de prévalence
        with torch.no_grad():
            self.head.classifier.bias.copy_(torch.from_numpy(log_prior))

    def forward(self, x_env, x_aux, x_l, x_b):
        embs  = [self.env_enc(x_env), self.aux_enc(x_aux),
                 self.landsat_enc(x_l), self.bioclim_enc(x_b)]
        fused = self.gating(embs)
        return self.head(fused)

# ══════════════════════════════════════════════════════════════
#  8. PERTE - AsymmetricLoss
#  gamma_neg=2 : moins agressif que 4, plus stable avec prior
#  Référence : Ridnik et al. 2021
# ══════════════════════════════════════════════════════════════
class ASL(nn.Module):
    def __init__(self, gp=0, gn=2, clip=0.05):
        super().__init__()
        self.gp, self.gn, self.clip = gp, gn, clip

    def forward(self, logits, y):
        p     = torch.sigmoid(logits)
        p_neg = (p + self.clip).clamp(max=1)
        lp    = (1 - p) ** self.gp * torch.log(p.clamp(1e-8))
        ln    = p_neg ** self.gn * torch.log((1 - p_neg).clamp(1e-8))
        return -(y * lp + (1 - y) * ln).mean()

# ══════════════════════════════════════════════════════════════
#  9. MÉTRIQUES ET SEUILLAGE
# ══════════════════════════════════════════════════════════════
def f1_sample(probs, targets, tau):
    preds = (probs > tau).float()
    tp = (preds * targets).sum(1)
    fp = (preds * (1 - targets)).sum(1)
    fn = ((1 - preds) * targets).sum(1)
    return (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()


def find_best_tau(probs, targets):
    best_f1, best_tau = 0., 0.15
    for tau in np.arange(0.05, 0.56, 0.01):
        f1 = f1_sample(probs, targets, float(tau))
        if f1 > best_f1:
            best_f1, best_tau = f1, round(float(tau), 2)
    return best_tau, best_f1


def make_adaptive_taus(sp_counts, tau_global, n_classes):
    """
    Seuil adaptatif par fréquence d'espèce.
    Espèces rares → τ plus bas (récupérer les vraies présences)
    Espèces communes → τ plus haut (éviter les faux positifs)
    Référence : Ung et al. 2023
    """
    taus = np.full(n_classes, tau_global, dtype=np.float32)
    taus[sp_counts > 500]  = min(tau_global + 0.08, 0.55)
    taus[sp_counts < 10]   = max(tau_global - 0.08, 0.05)
    return torch.tensor(taus, dtype=torch.float32)

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PIPELINE FINAL - GeoLifeCLEF MIASHS 2026")
    print("  Base : modèle collègue (0.20) + améliorations scientifiques")
    print("=" * 65)
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── 1. Labels ──────────────────────────────────────────
    print("\n[1/8] Chargement labels...")
    labels, species_list, sp2idx = load_labels(META_TRAIN)
    N_CLASSES = len(species_list)
    print(f"       {len(labels)} surveys | {N_CLASSES} espèces")

    # ── 2. Features tabulaires ─────────────────────────────
    print("\n[2/8] Features tabulaires (log1p + z-score + suppression redondances)...")
    env_raw  = load_env("train")
    test_raw = load_env("test")

    # Imputation par médiane train
    medians  = env_raw.median()
    env_raw  = env_raw.fillna(medians)
    test_raw = test_raw.fillna(medians)

    # Z-score (fit sur train uniquement)
    f_mean = env_raw.mean(); f_std = env_raw.std().replace(0, 1)
    env_norm  = ((env_raw  - f_mean) / f_std).astype(np.float32)
    test_norm = ((test_raw - f_mean) / f_std).fillna(0).astype(np.float32)

    ENV_DIM = env_norm.shape[1]
    print(f"       Features après suppression redondances : {ENV_DIM} (vs 64 avant)")

    # GPS - séparés en aux features (encodeur dédié)
    meta_tr = pd.read_csv(META_TRAIN)[["surveyId","lon","lat"]].drop_duplicates("surveyId").set_index("surveyId")
    meta_te = pd.read_csv(META_TEST )[["surveyId","lon","lat"]].drop_duplicates("surveyId").set_index("surveyId")
    gps_tr  = encode_gps(meta_tr)
    gps_te  = encode_gps(meta_te)

    env_train_dict = {sid: env_norm.loc[sid].values  for sid in env_norm.index}
    env_test_dict  = {sid: test_norm.loc[sid].values for sid in test_norm.index}
    aux_train_dict = {sid: gps_tr.loc[sid].values for sid in gps_tr.index if sid in env_train_dict}
    aux_test_dict  = {sid: gps_te.loc[sid].values for sid in gps_te.index if sid in env_test_dict}

    PlantDataset.env_dim = ENV_DIM

    # ── 3. Index séries temporelles ────────────────────────
    print("\n[3/8] Index séries temporelles...")
    l_tr = build_index(LANDSAT_TRAIN)
    l_te = build_index(LANDSAT_TEST)
    b_tr = build_index(BIO_TRAIN)
    b_te = build_index(BIO_TEST)
    print(f"       Landsat : {len(l_tr)} train | {len(l_te)} test")
    print(f"       Bioclim : {len(b_tr)} train | {len(b_te)} test")

    # ── 4. Stats normalisation z-score ─────────────────────
    print("\n[4/8] Calcul stats normalisation (z-score par bande/variable)...")
    l_mean, l_std = compute_stats(l_tr, dim_axis=0, n_dims=6,  n_sample=500, desc="Landsat")
    b_mean, b_std = compute_stats(b_tr, dim_axis=1, n_dims=19, n_sample=500, desc="Bioclim")

    print("       Landsat mean par bande:", [f"{x:.1f}" for x in l_mean.tolist()])
    print("       Landsat std  par bande:", [f"{x:.1f}" for x in l_std.tolist()])

    # ── 5. Split Leave-Region-Out ──────────────────────────
    print("\n[5/8] Split Leave-Region-Out...")
    meta_all = pd.read_csv(META_TRAIN)[["surveyId","region"]].drop_duplicates("surveyId")
    sid2reg  = dict(zip(meta_all.surveyId, meta_all.region))

    all_sids = [s for s in labels if s in env_train_dict]
    val_sids = [s for s in all_sids if sid2reg.get(s, "") in VAL_REGIONS]
    tr_sids  = [s for s in all_sids if sid2reg.get(s, "") not in VAL_REGIONS]

    print(f"       Train : {len(tr_sids)} surveys | régions : {sorted(set(sid2reg[s] for s in tr_sids))}")
    print(f"       Val   : {len(val_sids)} surveys | régions : {sorted(VAL_REGIONS)}")

    # Prior de fréquence par espèce
    sp_counts = np.zeros(N_CLASSES, dtype=np.float32)
    for sid in tr_sids:
        for sp in labels.get(sid, set()):
            sp_counts[sp] += 1
    prevalence = (sp_counts + 1e-6) / (len(tr_sids) + 1e-6)
    log_prior  = np.log(prevalence / (1 - prevalence + 1e-6)).astype(np.float32)
    MEDIAN_K   = int(np.median([len(labels[s]) for s in tr_sids]))
    print(f"       Médiane espèces/survey : {MEDIAN_K}")
    print(f"       Espèces rares (<10)    : {(sp_counts < 10).sum()}")
    print(f"       Espèces communes (>500): {(sp_counts > 500).sum()}")

    valid_test_ids = set(pd.read_csv(SAMPLE_SUB)["surveyId"])
    test_sids = sorted([s for s in env_test_dict if s in valid_test_ids])
    print(f"       Surveys test valides   : {len(test_sids)}")

    # ── 6. Datasets & DataLoaders ──────────────────────────
    def make_ds(sids, env_d, aux_d, li, bi, is_train):
        return PlantDataset(
            sids, env_d, aux_d, li, bi,
            l_mean, l_std, b_mean, b_std,
            labels=labels if is_train else None,
            n_classes=N_CLASSES,
        )

    train_ds = make_ds(tr_sids,   env_train_dict, aux_train_dict, l_tr, b_tr, True)
    val_ds   = make_ds(val_sids,  env_train_dict, aux_train_dict, l_tr, b_tr, True)
    test_ds  = make_ds(test_sids, env_test_dict,  aux_test_dict,  l_te, b_te, False)

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, **kw)
    print(f"\n       DataLoaders : train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # ── 7. Modèle ──────────────────────────────────────────
    print("\n[6/8] Construction du modèle...")
    model = MultimodalModel(
        env_dim=ENV_DIM, aux_dim=4,
        n_classes=N_CLASSES,
        log_prior=log_prior,
        d=64, dropout=0.3,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Paramètres : {n_params:,}")

    criterion = ASL(gn=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_dl),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # ── 8. Entraînement ────────────────────────────────────
    print(f"\n[7/8] Entraînement ({EPOCHS} epochs max, patience={PATIENCE})...")
    print("-" * 65)

    best_f1, best_tau, patience_cnt, best_ep = 0., 0.15, 0, 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss, t0 = 0., time.time()
        for x_env, x_aux, x_l, x_b, y in tqdm(train_dl, desc=f"Ep {epoch:3d}/{EPOCHS}", leave=False):
            x_env = x_env.to(DEVICE); x_aux = x_aux.to(DEVICE)
            x_l   = x_l.to(DEVICE);   x_b   = x_b.to(DEVICE)
            y     = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_env, x_aux, x_l, x_b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()

        # Validation
        model.eval()
        all_p, all_y, val_loss = [], [], 0.
        with torch.no_grad():
            for x_env, x_aux, x_l, x_b, y in val_dl:
                x_env = x_env.to(DEVICE); x_aux = x_aux.to(DEVICE)
                x_l   = x_l.to(DEVICE);   x_b   = x_b.to(DEVICE)
                y     = y.to(DEVICE)
                logits = model(x_env, x_aux, x_l, x_b)
                val_loss += criterion(logits, y).item()
                all_p.append(torch.sigmoid(logits).cpu())
                all_y.append(y.cpu())

        all_p = torch.cat(all_p); all_y = torch.cat(all_y)
        tau, f1 = find_best_tau(all_p, all_y)
        elapsed = time.time() - t0

        print(f"Ep {epoch:3d} | TrainL {tr_loss/len(train_dl):.4f} | "
              f"ValL {val_loss/len(val_dl):.4f} | "
              f"F1 {f1:.4f} (τ={tau:.2f}) | {elapsed:.0f}s")

        if f1 > best_f1:
            best_f1, best_tau, best_ep, patience_cnt = f1, tau, epoch, 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "f1": best_f1, "tau": best_tau,
                "species_list": species_list,
                "sp_counts": sp_counts,
                "f_mean": f_mean.values, "f_std": f_std.values,
                "medians": medians.values,
                "l_mean": l_mean, "l_std": l_std,
                "b_mean": b_mean, "b_std": b_std,
                "env_dim": ENV_DIM, "n_classes": N_CLASSES,
                "median_k": MEDIAN_K,
                "drop_features": list(DROP_FEATURES),
                "log_features": list(LOG_FEATURES),
            }, SAVE_DIR / "best_final.pt")
            print(f"  ✓ Sauvegardé (F1={best_f1:.4f}, τ={best_tau:.2f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n→ Early stopping ep {epoch} | Meilleur : ep{best_ep} F1={best_f1:.4f}")
                break

    print(f"\n{'='*65}")
    print(f"Meilleur F1 LRO : {best_f1:.4f}  τ={best_tau:.2f}  epoch={best_ep}")
    print(f"{'='*65}")

    # ── Soumission ─────────────────────────────────────────
    print("\n[8/8] Génération soumission...")
    ckpt = torch.load(SAVE_DIR / "best_final.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tau_g    = ckpt["tau"]
    sp_cts   = ckpt["sp_counts"]
    sp_list  = ckpt["species_list"]
    med_k    = ckpt["median_k"]
    taus     = make_adaptive_taus(sp_cts, tau_g, N_CLASSES)

    rows = []
    with torch.no_grad():
        for x_env, x_aux, x_l, x_b, sids in tqdm(test_dl, desc="Prédiction"):
            x_env = x_env.to(DEVICE); x_aux = x_aux.to(DEVICE)
            x_l   = x_l.to(DEVICE);   x_b   = x_b.to(DEVICE)
            probs = torch.sigmoid(model(x_env, x_aux, x_l, x_b)).cpu()

            for i, sid in enumerate(sids.tolist()):
                idx = (probs[i] > taus).nonzero(as_tuple=True)[0].tolist()
                if not idx:
                    idx = probs[i].topk(med_k).indices.tolist()
                sp_ids = sorted([sp_list[j] for j in idx])
                rows.append({"surveyId": sid, "predictions": " ".join(map(str, sp_ids))})

    sub = pd.DataFrame(rows)
    assert len(sub) == 14784, f"ERREUR : {len(sub)} lignes au lieu de 14784"
    assert set(sub.surveyId) == valid_test_ids, "ERREUR : IDs incorrects"

    out = SAVE_DIR / "submission_final.csv"
    sub.to_csv(out, index=False)
    n_sp = sub["predictions"].apply(lambda x: len(x.split())).mean()
    print(f"\n✓ Soumission : {out}")
    print(f"  Surveys : {len(sub)} | Espèces moyennes : {n_sp:.1f} (cible : {med_k})")
    print(sub.head(5).to_string())


if __name__ == "__main__":
    main()