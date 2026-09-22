"""
Branche Environnementale - EnvironmentalMLP
============================================

Entree : variables scalaires environnementales
  bioclim  (19) : moyennes bioclimatiques CHELSA 1981-2010
  soilgrids (9) : proprietes du sol (pH, argile, carbone, ...)
  --> total : 28 features par defaut

Sortie : (n_species,) logits - BCEWithLogitsLoss

Architecture :
  Input(28) → BN
  → Linear(28, 512) → BN → ReLU → Dropout(0.3)
  → Linear(512, 512) → BN → ReLU → Dropout(0.3)
  → Linear(512, 256) → BN → ReLU → Dropout(0.2)
  → Linear(256, n_species)

Choix :
- BatchNorm avant ReLU : stabilise l'entrainement meme avec des features
  tres heterogenes (pH 4-9, carbone 0-300, elevation -100-4000 ...)
- Dropout decroissant (0.3 → 0.3 → 0.2) : regularisation forte en debut,
  plus legere pres de la tete de classification
- 3 couches cachees : suffisant pour capturer les interactions non-lineaires
  entre variables climatiques et pedologiques
"""

import torch
import torch.nn as nn


class EnvironmentalMLP(nn.Module):
    """
    MLP pour la prediction multi-label d'especes a partir de variables
    environnementales (bioclim + soilgrids).

    Args:
        n_features : nombre de features d'entree (bioclim + soilgrids)
        n_species  : nombre d'especes cibles (sortie)
        hidden     : tailles des 3 couches cachees
        dropouts   : taux de dropout pour chaque couche cachee
    """

    def __init__(
        self,
        n_features=28,
        n_species=10_000,
        hidden=(512, 512, 256),
        dropouts=(0.3, 0.3, 0.2),
    ):
        super().__init__()

        assert len(hidden) == 3, "hidden doit avoir exactement 3 elements"
        assert len(dropouts) == 3, "dropouts doit avoir exactement 3 elements"

        h1, h2, h3 = hidden
        d1, d2, d3 = dropouts

        # Normalisation de l'entree (complement au Z-score externe)
        self.input_bn = nn.BatchNorm1d(n_features)

        # Couche 1
        self.layer1 = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(d1),
        )

        # Couche 2
        self.layer2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(d2),
        )

        # Couche 3
        self.layer3 = nn.Sequential(
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(d3),
        )

        # Tete de classification
        self.head = nn.Linear(h3, n_species)

        self._init_weights()

    def _init_weights(self):
        """Initialisation He (kaiming) pour les couches lineaires avec ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x : (B, n_features)
        x = self.input_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)   # (B, n_species) logits


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B          = 8
    n_features = 28   # 19 bioclim + 9 soilgrids
    n_species  = 10_000

    model  = EnvironmentalMLP(n_features=n_features, n_species=n_species)
    n_pars = sum(p.numel() for p in model.parameters())
    print("Parametres : {:,}".format(n_pars))

    x   = torch.randn(B, n_features)
    out = model(x)
    print("Input  : {}".format(x.shape))
    print("Output : {}".format(out.shape))   # (8, 10000)
