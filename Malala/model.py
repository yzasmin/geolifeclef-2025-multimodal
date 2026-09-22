"""
Modele multimodal - Species Distribution Model (PA only)
=========================================================

Inspire de :
  - MultimodalEnsemble (Picek & Larcher, GeoLifeCLEF2024) : ResNet/Swin preentraine
  - Chen et al. GeoLifeCLEF2024 : branche bioclim separee
  - Adrian/model.py : GELU, LayerNorm, Transformer pour series temporelles,
                      branch_dim uniforme, pre-norm Transformer, image optionnelle

Entrees par surveyId :
  image      : (4, 64, 64)   Sentinel-2 [R, G, B, NIR]          - optionnel
  landsat    : (B, 21, 24)   serie Landsat (21 ans, 24 features) - optionnel
  bioclim    : (19,)         moyennes CHELSA 1981-2010
  env        : (N_ENV,)      sol(9)+elev(1)+landcover(1)+footprint(22)

Sorties : (n_species,) logits

Architecture :
  Sentinel-2  → ResNet18(4ch, ImageNet)   → branch_dim   [optionnel]
  Landsat     → TransformerEncoder        → branch_dim   [optionnel]
  Bioclim     → MLPEncoder(LayerNorm+GELU)→ branch_dim
  Env         → MLPEncoder(LayerNorm+GELU)→ branch_dim
  Fusion      → cat → Linear → GELU → LayerNorm → Dropout → n_species

Choix cles vs baseline :
  GELU         : meilleure regularisation implicite que ReLU (gradient non nul pour x<0)
  LayerNorm    : plus stable que BatchNorm pour donnees tabulaires (independant du batch size)
  Transformer  : modelise les dependances temporelles inter-saisons dans Landsat
  pre-norm     : norm_first=True → gradients plus stables en debut d'entrainement
  branch_dim   : embedding uniforme (256) pour toutes les modalites → fusion equilibree
"""

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def setup_device(gpu_id=0):
    # type: (Union[int, str]) -> torch.device
    gpu_id = os.environ.get("GPU_ID", gpu_id)

    if gpu_id == "cpu" or not torch.cuda.is_available():
        print("Running on CPU.")
        return torch.device("cpu")

    gpu_id = int(gpu_id)
    n_gpus = torch.cuda.device_count()
    if gpu_id >= n_gpus:
        raise ValueError("GPU {} demande mais {} disponible(s).".format(gpu_id, n_gpus))

    device = torch.device("cuda:{}".format(gpu_id))
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print("GPU {}: {}  ({:.1f} GB)".format(gpu_id, props.name, props.total_memory / 1024**3))
    return device


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
N_SPECIES  = 10_000
N_BIOCLIM  = 19    # CHELSA 1981-2010
N_ENV      = 33    # sol(9) + elev(1) + landcover(1) + footprint(22)
BRANCH_DIM = 256   # dimension uniforme de sortie pour chaque encodeur


# ---------------------------------------------------------------------------
# 1. Encodeur Sentinel-2 : ResNet18 preentraine, 4 canaux
#    Avantage vs Adrian : poids ImageNet → convergence 2-3x plus rapide
# ---------------------------------------------------------------------------
class SentinelEncoder(nn.Module):
    """
    ResNet18 adapte pour 4 canaux Sentinel-2 (R, G, B, NIR).

    Le canal NIR est initialise comme la moyenne des poids RGB pour
    exploiter les features visuelles preapprises sur ImageNet.
    Sortie : (B, branch_dim)
    """

    def __init__(self, pretrained=True, branch_dim=BRANCH_DIM):
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            base = tvm.resnet18(weights=weights)
        except TypeError:
            base = tvm.resnet18(pretrained=pretrained)

        # 3 canaux → 4 canaux
        old = base.conv1
        new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:4] = old.weight.mean(dim=1, keepdim=True)
        base.conv1 = new

        # Projection vers branch_dim avec GELU (inspire d'Adrian)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, branch_dim),
            nn.GELU(),
        )
        self.net = base

    def forward(self, x):
        # x : (B, 4, 64, 64)
        return self.net(x)   # (B, branch_dim)


# ---------------------------------------------------------------------------
# 2. Encodeur Landsat : Transformer sur serie temporelle
#    Inspire d'Adrian : modelise les dependances inter-saisons
#
#    Input reshape : (B, 6, 4, 21) → (B, 21, 6*4) = (B, 21, 24)
#    21 pas de temps (annees), 24 features par pas (6 bandes × 4 saisons)
# ---------------------------------------------------------------------------
class LandsatTransformerEncoder(nn.Module):
    """
    Transformer encoder pour la serie temporelle Landsat.

    Pourquoi Transformer vs ResNet18 2D (notre baseline) ?
    - Modelise explicitement les relations temporelles entre annees
    - Attention multi-tetes : pondere les annees les plus informatives
    - pre-norm (norm_first=True) : gradients plus stables (inspire d'Adrian)

    Format entree attendu : (B, 6, 4, 21)
    Reshape interne    → (B, 21, 24)  [21 annees, 24=6bandes×4saisons]
    Sortie             → (B, branch_dim)
    """

    def __init__(
        self,
        landsat_shape=(6, 4, 21),   # (bandes, saisons, annees)
        branch_dim=BRANCH_DIM,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        # n_timesteps = annees, in_features = bandes × saisons
        self.n_timesteps = landsat_shape[2]              # 21
        self.in_features = landsat_shape[0] * landsat_shape[1]   # 6 × 4 = 24

        # Normalisation de l'entree
        self.input_norm = nn.LayerNorm(self.in_features)

        # Projection vers branch_dim
        self.input_proj = nn.Linear(self.in_features, branch_dim)

        # Transformer (pre-norm = norm_first=True, inspire d'Adrian)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=branch_dim,
            nhead=n_heads,
            dim_feedforward=branch_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm : plus stable en debut d'entrainement
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm    = nn.LayerNorm(branch_dim)

    def forward(self, x):
        # x : (B, 6, 4, 21)
        B = x.shape[0]
        # Reshape : (B, 6, 4, 21) → (B, 21, 24)
        x = x.permute(0, 3, 1, 2).reshape(B, self.n_timesteps, self.in_features)
        x = self.input_norm(x)
        x = self.input_proj(x)              # (B, 21, branch_dim)
        x = self.transformer(x)             # (B, 21, branch_dim)
        x = self.out_norm(x)
        return x.mean(dim=1)               # (B, branch_dim) - moyenne temporelle


# ---------------------------------------------------------------------------
# 3. Encodeur MLP generique (bioclim + env)
#    Inspire d'Adrian : LayerNorm + GELU (plus stable que BatchNorm+ReLU
#    pour les donnees tabulaires, independant de la taille de batch)
# ---------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    """
    MLP avec LayerNorm et GELU pour les features tabulaires.

    LayerNorm vs BatchNorm pour les scalaires env :
    - BatchNorm depend de la statistique du batch → instable si batch petit
    - LayerNorm normalise par feature → stable meme avec batch_size=1
    - GELU : gradient non nul pour x<0, meilleure regularisation implicite

    Sortie : (B, branch_dim)
    """

    def __init__(
        self,
        in_features,
        branch_dim=BRANCH_DIM,
        hidden_dims=(256, 256),
        dropout=0.1,
    ):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, branch_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x : (B, in_features)
        return self.net(x)   # (B, branch_dim)


# ---------------------------------------------------------------------------
# 4. Tete de fusion
#    Inspire d'Adrian : LayerNorm + GELU (vs notre ReLU+Dropout)
# ---------------------------------------------------------------------------
class FusionHead(nn.Module):
    """
    Fusion des embeddings de toutes les branches + classification.

    Utilise GELU + LayerNorm (inspire d'Adrian) au lieu de ReLU+Dropout
    pour une meilleure stabilite et regularisation.
    """

    def __init__(self, fuse_in, n_species, hidden=512, dropout=0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, n_species)

    def forward(self, x):
        return self.classifier(self.fusion(x))


# ---------------------------------------------------------------------------
# 5. Modele complet
# ---------------------------------------------------------------------------
class PlantModel(nn.Module):
    """
    Modele multimodal SDM - meilleur des deux approches.

    Branches actives selon les donnees disponibles :
      use_image   : Sentinel-2 (ResNet18 preentraine)
      use_landsat : serie temporelle Landsat (Transformer)
      bioclim     : toujours active
      env         : toujours active

    Avantages vs baseline (Malala/model.py v1) :
      + Transformer Landsat : modelise les dependances temporelles
      + GELU + LayerNorm   : stabilite et regularisation ameliorees
      + branch_dim uniforme : toutes les branches contribuent de facon egale
      + Image optionnelle  : peut tourner sans Sentinel si TIFF manquants

    Avantages vs Adrian/model.py :
      + ResNet18 preentraine ImageNet pour Sentinel → convergence 2-3x plus rapide
      + Branche bioclim separee (Chen et al.) → representation climatique dedicee

    Args:
        n_species     : nombre d'especes cibles
        n_bioclim     : features bioclimatiques (19)
        n_env         : autres features env (33)
        landsat_shape : shape du cube Landsat (6, 4, 21)
        branch_dim    : dimension uniforme des embeddings (256)
        use_image     : activer la branche Sentinel-2
        use_landsat   : activer la branche Landsat
        pretrained    : poids ImageNet pour ResNet18
    """

    def __init__(
        self,
        n_species=N_SPECIES,
        n_bioclim=N_BIOCLIM,
        n_env=N_ENV,
        landsat_shape=(6, 4, 21),
        branch_dim=BRANCH_DIM,
        use_image=True,
        use_landsat=True,
        pretrained=True,
        dropout=0.2,
    ):
        super().__init__()
        self.use_image   = use_image
        self.use_landsat = use_landsat

        # Branches conditionnelles
        if use_image:
            self.sentinel_enc = SentinelEncoder(pretrained=pretrained, branch_dim=branch_dim)
        if use_landsat:
            self.landsat_enc = LandsatTransformerEncoder(
                landsat_shape=landsat_shape, branch_dim=branch_dim, dropout=dropout)

        # Branches toujours actives
        self.bioclim_enc = MLPEncoder(n_bioclim, branch_dim=branch_dim,
                                      hidden_dims=(256, 256), dropout=dropout)
        self.env_enc     = MLPEncoder(n_env, branch_dim=branch_dim,
                                      hidden_dims=(256, 256), dropout=dropout)

        # Dimension de fusion
        n_branches = 2 + int(use_image) + int(use_landsat)
        fuse_in    = n_branches * branch_dim

        self.head = FusionHead(fuse_in, n_species, hidden=branch_dim * 2, dropout=dropout)

    def forward(self, bioclim, env, image=None, landsat=None):
        # type: (torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]) -> torch.Tensor
        embeddings = []

        if self.use_image:
            if image is None:
                raise ValueError("use_image=True mais image=None")
            embeddings.append(self.sentinel_enc(image))

        if self.use_landsat:
            if landsat is None:
                raise ValueError("use_landsat=True mais landsat=None")
            embeddings.append(self.landsat_enc(landsat))

        embeddings.append(self.bioclim_enc(bioclim))
        embeddings.append(self.env_enc(env))

        h = torch.cat(embeddings, dim=1)
        return self.head(h)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = setup_device(0)
    B      = 4

    # Mode complet (toutes les modalites)
    model = PlantModel(
        n_species=N_SPECIES, n_bioclim=N_BIOCLIM, n_env=N_ENV,
        use_image=True, use_landsat=True,
    ).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print("Complet   | Parametres : {:,}".format(n_p))

    out = model(
        image   =torch.randn(B, 4, 64, 64).to(device),
        landsat =torch.randn(B, 6, 4, 21).to(device),
        bioclim =torch.randn(B, N_BIOCLIM).to(device),
        env     =torch.randn(B, N_ENV).to(device),
    )
    print("Complet   | Output : {}".format(out.shape))   # (4, 10000)

    # Mode env seulement (pas d'images, pas de Landsat)
    model_env = PlantModel(
        n_species=N_SPECIES, n_bioclim=N_BIOCLIM, n_env=N_ENV,
        use_image=False, use_landsat=False,
    ).to(device)
    n_p2 = sum(p.numel() for p in model_env.parameters())
    print("Env only  | Parametres : {:,}".format(n_p2))

    out2 = model_env(
        bioclim=torch.randn(B, N_BIOCLIM).to(device),
        env    =torch.randn(B, N_ENV).to(device),
    )
    print("Env only  | Output : {}".format(out2.shape))
