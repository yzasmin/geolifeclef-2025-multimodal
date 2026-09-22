"""
Framework d'explication multimodal - model-agnostic
=====================================================

Compatible avec n'importe quel modele PyTorch qui accepte des tenseurs
en entree, via un systeme d'adaptateur (ModelAdapter).

Adaptateurs fournis :
  - PlantModelAdapter   : pour Malala/model.py (PlantModel)
  - AdrianModelAdapter  : pour Adrian/model.py (MultimodalPlantModel)
  - GenericAdapter      : pour tout autre modele, en passant une fn forward custom

Methodes d'explication :
  A. GradCAM         : zones importantes dans l'image Sentinel-2 (via Captum)
  B. SHAP            : importance des features tabulaires (bioclim + env)
  C. Attention maps  : annees les plus pertinentes pour Landsat (Transformer)

Installation des dependances optionnelles :
  pip install captum shap matplotlib seaborn

Usage minimal :
    from explainability import explain
    explain(model, adapter="adrian", val_loader=loader, output_dir="./xai")
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")   # sans affichage GUI (serveur)
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARN: matplotlib/seaborn non installe - pas de visualisation")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARN: shap non installe  (pip install shap)")

try:
    from captum.attr import LayerGradCam
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("WARN: captum non installe (pip install captum)")


# =============================================================================
# ADAPTATEURS : traduisent un batch en appel forward() du modele
# =============================================================================

class ModelAdapter:
    """
    Interface de base pour rendre le framework model-agnostic.

    Sous-classer et implementer :
      - forward(batch, model, device) -> logits (B, n_species)
      - get_image(batch)    -> Tensor (B, 4, 64, 64) ou None
      - get_landsat(batch)  -> Tensor (B, 6, 4, 21)  ou None
      - get_tabular(batch)  -> Tensor (B, n_features) [bioclim + env concatenes]
      - get_image_encoder(model) -> nn.Module de l'encodeur image, ou None
    """

    def forward(self, batch, model, device):
        # type: (object, nn.Module, torch.device) -> torch.Tensor
        raise NotImplementedError

    def get_image(self, batch):
        # type: (object) -> Optional[torch.Tensor]
        return None

    def get_landsat(self, batch):
        # type: (object) -> Optional[torch.Tensor]
        return None

    def get_tabular(self, batch):
        # type: (object) -> Optional[torch.Tensor]
        """Retourne bioclim + env concatenes pour SHAP."""
        return None

    def get_labels(self, batch):
        # type: (object) -> Optional[torch.Tensor]
        return None

    def get_image_encoder(self, model):
        # type: (nn.Module) -> Optional[nn.Module]
        """Retourne le sous-module encodeur image pour GradCAM."""
        return None

    def get_last_conv_layer(self, model):
        # type: (nn.Module) -> Optional[nn.Module]
        """Retourne la derniere couche de convolution pour GradCAM."""
        return None


class PlantModelAdapter(ModelAdapter):
    """
    Adaptateur pour Malala/model.py : PlantModel.

    Batch attendu (dict) :
      batch["image"]   : (B, 4, 64, 64)
      batch["landsat"] : (B, 6, 4, 21)
      batch["bioclim"] : (B, 19)
      batch["env"]     : (B, 33)
      batch["label"]   : (B, n_species)
    """

    def forward(self, batch, model, device):
        kw = {
            "bioclim": batch["bioclim"].to(device),
            "env":     batch["env"].to(device),
        }
        if model.use_image and "image" in batch:
            kw["image"] = batch["image"].to(device)
        if model.use_landsat and "landsat" in batch:
            kw["landsat"] = batch["landsat"].to(device)
        return model(**kw)

    def get_image(self, batch):
        return batch.get("image")

    def get_landsat(self, batch):
        return batch.get("landsat")

    def get_tabular(self, batch):
        parts = []
        if "bioclim" in batch:
            parts.append(batch["bioclim"])
        if "env" in batch:
            parts.append(batch["env"])
        return torch.cat(parts, dim=1) if parts else None

    def get_labels(self, batch):
        return batch.get("label")

    def get_image_encoder(self, model):
        return getattr(model, "sentinel_enc", None)

    def get_last_conv_layer(self, model):
        enc = self.get_image_encoder(model)
        if enc is None:
            return None
        # ResNet18 : derniere couche de layer4
        try:
            return enc.net.layer4[-1]
        except (AttributeError, IndexError):
            return None


class AdrianModelAdapter(ModelAdapter):
    """
    Adaptateur pour Adrian/model.py : MultimodalPlantModel.

    Batch attendu (tuple) :
      (images, landsat_series, climate_series, env_features, aux_features)
      ou sous-ensemble selon ce que le DataLoader produit.

    Structure du modele Adrian :
      forward(env_features, aux_features, landsat_series, climate_series, image=None)
    """

    def __init__(self, n_bioclim=19):
        # type: (int) -> None
        self.n_bioclim = n_bioclim

    def _unpack(self, batch):
        # type: (object) -> dict
        """Depaquette le batch Adrian (tuple ou dict) en dict normalise."""
        if isinstance(batch, dict):
            return batch

        # Tuple : (env, aux, landsat, climate, image, targets) - ordre de collate_fn Adrian
        keys = ["env", "aux", "landsat", "climate", "image", "label"]
        result = {}
        if isinstance(batch, (list, tuple)):
            for i, val in enumerate(batch):
                if i < len(keys):
                    result[keys[i]] = val
        return result

    def forward(self, batch, model, device):
        b = self._unpack(batch)

        # Arguments obligatoires
        env = b.get("env")
        aux = b.get("aux")

        if env is None or aux is None:
            raise ValueError(
                "AdrianModelAdapter : batch doit contenir 'env' et 'aux'. "
                "Recu : {}".format(list(b.keys()))
            )

        kw = {
            "env_features": env.to(device),
            "aux_features": aux.to(device),
        }

        # Series temporelles (requises par MultimodalPlantModel.forward)
        landsat = b.get("landsat")
        climate = b.get("climate")
        if landsat is None:
            raise ValueError(
                "AdrianModelAdapter : 'landsat' manquant dans le batch "
                "(requis par MultimodalPlantModel.forward). Cles presentes : {}".format(list(b.keys()))
            )
        if climate is None:
            raise ValueError(
                "AdrianModelAdapter : 'climate' manquant dans le batch "
                "(requis par MultimodalPlantModel.forward). Cles presentes : {}".format(list(b.keys()))
            )
        kw["landsat_series"] = landsat.to(device)
        kw["climate_series"] = climate.to(device)

        # Image (optionnelle selon use_image_branch)
        if getattr(model, "use_image_branch", False):
            image = b.get("image")
            if image is None:
                raise ValueError(
                    "AdrianModelAdapter : use_image_branch=True mais 'image' manquant dans le batch."
                )
            kw["image"] = image.to(device)

        return model(**kw)

    def get_image(self, batch):
        b = self._unpack(batch)
        return b.get("image")

    def get_landsat(self, batch):
        b = self._unpack(batch)
        return b.get("landsat")

    def get_tabular(self, batch):
        b = self._unpack(batch)
        parts = []
        for key in ("env", "aux"):
            if key in b and b[key] is not None:
                parts.append(b[key])
        return torch.cat(parts, dim=1) if parts else None

    def get_labels(self, batch):
        b = self._unpack(batch)
        return b.get("label")

    def get_image_encoder(self, model):
        return getattr(model, "image_encoder", None)

    def get_last_conv_layer(self, model):
        enc = self.get_image_encoder(model)
        if enc is None:
            return None
        # ImageEncoder Adrian : Sequential([ConvBlock×4, AvgPool, Flatten, Linear])
        # enc.encoder[-1]=Linear, [-2]=Flatten, [-3]=AvgPool, [-4]=last ConvBlock
        try:
            return enc.encoder[-4]
        except (AttributeError, IndexError):
            return None


class GenericAdapter(ModelAdapter):
    """
    Adaptateur generique pour tout modele PyTorch.
    Passe une fonction forward_fn custom.

    Usage :
        adapter = GenericAdapter(
            forward_fn=lambda batch, model, device: model(batch["x"].to(device)),
            get_image_fn=lambda batch: batch.get("image"),
        )
    """

    def __init__(
        self,
        forward_fn,          # type: Callable
        get_image_fn=None,   # type: Optional[Callable]
        get_landsat_fn=None, # type: Optional[Callable]
        get_tabular_fn=None, # type: Optional[Callable]
        get_labels_fn=None,  # type: Optional[Callable]
        last_conv_layer=None,
    ):
        self._forward    = forward_fn
        self._get_image  = get_image_fn
        self._get_landsat = get_landsat_fn
        self._get_tabular = get_tabular_fn
        self._get_labels  = get_labels_fn
        self._conv_layer  = last_conv_layer

    def forward(self, batch, model, device):
        return self._forward(batch, model, device)

    def get_image(self, batch):
        return self._get_image(batch) if self._get_image else None

    def get_landsat(self, batch):
        return self._get_landsat(batch) if self._get_landsat else None

    def get_tabular(self, batch):
        return self._get_tabular(batch) if self._get_tabular else None

    def get_labels(self, batch):
        return self._get_labels(batch) if self._get_labels else None

    def get_last_conv_layer(self, model):
        return self._conv_layer


def get_adapter(name, **kwargs):
    # type: (str, ...) -> ModelAdapter
    """
    Raccourci pour obtenir un adaptateur par nom.

    Usage :
        adapter = get_adapter("malala")
        adapter = get_adapter("adrian")
        adapter = get_adapter("generic", forward_fn=my_fn)
    """
    name = name.lower()
    if name in ("malala", "plant", "plantmodel"):
        return PlantModelAdapter()
    if name in ("adrian", "multimodal", "adrianmodel"):
        return AdrianModelAdapter(**kwargs)
    if name == "generic":
        return GenericAdapter(**kwargs)
    raise ValueError("Adapter inconnu : '{}'. Choix : malala, adrian, generic".format(name))


# =============================================================================
# A. GRADCAM - zones importantes dans l'image Sentinel-2
# =============================================================================

def gradcam(
    model,          # type: nn.Module
    adapter,        # type: ModelAdapter
    batch,
    species_idx=0,  # type: int
    device="cuda",  # type: str
    output_path="gradcam.png",  # type: str
):
    # type: (...) -> Optional[np.ndarray]
    """
    GradCAM sur la branche image Sentinel-2.

    Retourne la carte d'attribution (B, H, W) ou None si Captum absent.
    """
    if not HAS_CAPTUM:
        print("SKIP GradCAM : captum non installe (pip install captum)")
        return None

    images = adapter.get_image(batch)
    if images is None:
        print("SKIP GradCAM : pas d'image dans ce batch")
        return None

    conv_layer = adapter.get_last_conv_layer(model)
    if conv_layer is None:
        print("SKIP GradCAM : couche de convolution non trouvee")
        return None

    model.eval()
    images = images.to(device)

    # Captum appelle forward(image) uniquement - on encapsule le modele
    # dans un nn.Module qui fixe toutes les autres entrees en closure.
    class _ImageOnlyWrapper(nn.Module):
        def __init__(self, model, adapter, batch, device):
            super().__init__()
            self.model   = model
            self.adapter = adapter
            self.batch   = batch
            self.device  = device

        def forward(self, img):
            # Depaquette toujours en dict pour remplacer "image" sans ambiguite
            # (evite l'erreur index pour les tuples Adrian ou l'ordre n'est pas image en 0)
            if hasattr(self.adapter, "_unpack"):
                b2 = dict(self.adapter._unpack(self.batch))
            elif isinstance(self.batch, dict):
                b2 = dict(self.batch)
            else:
                b2 = {}
            b2 = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b2.items()}
            b2["image"] = img
            return self.adapter.forward(b2, self.model, self.device)

    wrapper    = _ImageOnlyWrapper(model, adapter, batch, device)
    conv_layer = adapter.get_last_conv_layer(wrapper.model)

    lgc  = LayerGradCam(wrapper, conv_layer)
    attr = lgc.attribute(images, target=species_idx)   # (B, 1, H', W')

    if not HAS_MPL:
        return attr.detach().cpu().numpy()

    n = min(images.shape[0], 4)
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axs = axs.reshape(1, -1)

    for i in range(n):
        # RGB normalise
        rgb = images[i, :3].cpu().permute(1, 2, 0).numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

        # Attribution upsampled vers 64x64
        a = attr[i, 0].cpu().detach().numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
        # Resize vers (64,64) si necessaire
        if a.shape != (64, 64):
            from PIL import Image as PILImg
            a = np.array(PILImg.fromarray((a * 255).astype(np.uint8)).resize((64, 64))) / 255.0

        axs[i, 0].imshow(rgb)
        axs[i, 0].set_title("Sentinel-2 RGB")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(a, cmap="hot")
        axs[i, 1].set_title("GradCAM (espece {})".format(species_idx))
        axs[i, 1].axis("off")

        axs[i, 2].imshow(rgb)
        axs[i, 2].imshow(a, cmap="hot", alpha=0.45)
        axs[i, 2].set_title("Overlay")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("GradCAM -> {}".format(output_path))
    return attr.detach().cpu().numpy()


# =============================================================================
# B. SHAP - importance des features tabulaires
# =============================================================================

def shap_tabular(
    model,                       # type: nn.Module
    adapter,                     # type: ModelAdapter
    batches,                     # type: list  - liste de batch pour constituer background
    feature_names=None,          # type: Optional[List[str]]
    n_background=50,             # type: int
    n_explain=20,                # type: int
    device="cuda",               # type: str
    output_dir=".",              # type: str
):
    # type: (...) -> Optional[Dict]
    """
    SHAP KernelExplainer sur les features tabulaires (bioclim + env).

    n_background : taille de l'ensemble de reference (plus grand = plus precis mais plus lent)
    n_explain    : nombre d'exemples a expliquer

    Retourne : {"shap_values", "feature_names", "feature_importance"} ou None
    """
    if not HAS_SHAP:
        print("SKIP SHAP : shap non installe (pip install shap)")
        return None

    model.eval()

    # Collecte les features tabulaires de tous les batches
    all_tab = []
    for b in batches:
        t = adapter.get_tabular(b)
        if t is not None:
            all_tab.append(t.cpu().numpy())
    if not all_tab:
        print("SKIP SHAP : pas de features tabulaires dans les batches")
        return None

    data = np.concatenate(all_tab, axis=0)   # (N, n_features)
    n_feats = data.shape[1]

    if feature_names is None:
        feature_names = ["feat_{}".format(i) for i in range(n_feats)]
    # Truncate/extend si necessaire
    feature_names = list(feature_names)[:n_feats]
    while len(feature_names) < n_feats:
        feature_names.append("feat_{}".format(len(feature_names)))

    # Pre-calcule les tenseurs fixes depuis le batch de reference
    # (tout ce qui n'est PAS les features tabulaires)
    ref_batch = batches[0]
    ref_b     = adapter._unpack(ref_batch) if isinstance(adapter, AdrianModelAdapter) else ref_batch

    # Pour Adrian : recupere env_dim pour le split env/aux
    if isinstance(adapter, AdrianModelAdapter) and "env" in ref_b:
        _env_dim_shap = ref_b["env"].shape[1]
    else:
        _env_dim_shap = None

    # Tenseurs fixes (image, landsat, climate) - on prend la premiere ligne et on repete
    def _get_fixed(key, ref):
        t = ref.get(key) if isinstance(ref, dict) else None
        return t[0:1].to(device) if t is not None else None

    _fixed_image   = _get_fixed("image",   ref_b)
    _fixed_landsat = _get_fixed("landsat", ref_b)
    _fixed_climate = _get_fixed("climate", ref_b)

    # Pour Adrian, landsat et climate sont requis (args non optionnels dans forward)
    if isinstance(adapter, AdrianModelAdapter):
        if _fixed_landsat is None:
            print("SKIP SHAP : Adrian requiert 'landsat' dans le batch")
            return None
        if _fixed_climate is None:
            print("SKIP SHAP : Adrian requiert 'climate' dans le batch")
            return None

    def _predict(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
        B   = x_t.shape[0]

        if isinstance(adapter, PlantModelAdapter):
            n_bio = 19
            fake_batch = {
                "bioclim": x_t[:, :n_bio],
                "env":     x_t[:, n_bio:],
            }
            if _fixed_image is not None and hasattr(model, "use_image") and model.use_image:
                fake_batch["image"]   = _fixed_image.expand(B, -1, -1, -1)
            if _fixed_landsat is not None and hasattr(model, "use_landsat") and model.use_landsat:
                fake_batch["landsat"] = _fixed_landsat.expand(B, -1, -1, -1)

        elif isinstance(adapter, AdrianModelAdapter):
            ed = _env_dim_shap or (x_t.shape[1] // 2)
            fake_batch = {
                "env": x_t[:, :ed],
                "aux": x_t[:, ed:],
            }
            # Toujours fournir landsat, climate et image (meme fixes) pour eviter ValueError
            if _fixed_landsat is not None:
                fake_batch["landsat"] = _fixed_landsat.expand(B, -1, -1)
            if _fixed_climate is not None:
                fake_batch["climate"] = _fixed_climate.expand(B, -1, -1)
            if _fixed_image is not None:
                fake_batch["image"]   = _fixed_image.expand(B, -1, -1, -1)

        else:
            fake_batch = x_t

        with torch.no_grad():
            logits = adapter.forward(fake_batch, model, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs.mean(axis=1, keepdims=True)   # (B, 1)

    # Background et exemples a expliquer
    bg_idx  = np.random.choice(len(data), size=min(n_background, len(data)), replace=False)
    ex_idx  = np.random.choice(len(data), size=min(n_explain,    len(data)), replace=False)
    bg_data = data[bg_idx]
    ex_data = data[ex_idx]

    print("SHAP : {} background, {} a expliquer, {} features...".format(
        len(bg_data), len(ex_data), n_feats))

    explainer   = shap.KernelExplainer(_predict, bg_data)
    shap_values = explainer.shap_values(ex_data, nsamples=100)   # (n_explain, n_feats)

    if isinstance(shap_values, list):
        sv = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        sv = np.abs(shap_values)

    feat_imp = sv.mean(axis=0)   # (n_feats,)

    if HAS_MPL:
        os.makedirs(output_dir, exist_ok=True)

        # Graphique 1 : importances globales (top 20)
        top_n  = min(20, n_feats)
        top_idx = np.argsort(feat_imp)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), feat_imp[top_idx], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Importance des features (top {})".format(top_n))
        plt.tight_layout()
        path1 = os.path.join(output_dir, "shap_importance.png")
        plt.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close()
        print("SHAP importance -> {}".format(path1))

        # Graphique 2 : dependance pour les 4 features les plus importantes
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        for k, fi in enumerate(top_idx[-4:]):
            ax = axs[k // 2, k % 2]
            ax.scatter(ex_data[:, fi], sv[:, fi], alpha=0.6, s=20)
            ax.set_xlabel(feature_names[fi], fontsize=9)
            ax.set_ylabel("SHAP value")
            ax.set_title("Dependance : {}".format(feature_names[fi]))
        plt.tight_layout()
        path2 = os.path.join(output_dir, "shap_dependence.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        print("SHAP dependance -> {}".format(path2))

    return {
        "shap_values":        sv,
        "feature_names":      feature_names,
        "feature_importance": feat_imp,
    }


# =============================================================================
# C. ATTENTION MAP - series temporelles Landsat
# =============================================================================

def landsat_attention(
    model,                            # type: nn.Module
    adapter,                          # type: ModelAdapter
    batch,
    output_path="landsat_attention.png",  # type: str
    device="cuda",                    # type: str
    year_labels=None,                 # type: Optional[List[str]]
):
    # type: (...) -> Optional[np.ndarray]
    """
    Visualise les poids d'attention du Transformer Landsat.
    Montre quelles annees le modele trouve les plus informatives.

    Retourne la matrice d'attention moyennee sur les tetes (n_years, n_years).
    """
    landsat_data = adapter.get_landsat(batch)
    if landsat_data is None:
        print("SKIP attention : pas de Landsat dans ce batch")
        return None

    # Trouve l'encodeur Landsat dans le modele
    enc = None
    for name in ("landsat_enc", "landsat_encoder"):
        enc = getattr(model, name, None)
        if enc is not None:
            break
    if enc is None:
        print("SKIP attention : encodeur Landsat non trouve dans le modele")
        return None

    # Trouve le Transformer dans l'encodeur
    transformer = None
    for name in ("transformer", "encoder"):
        transformer = getattr(enc, name, None)
        if transformer is not None:
            break
    if transformer is None:
        print("SKIP attention : Transformer non trouve dans l'encodeur Landsat")
        return None

    model.eval()
    landsat_data = landsat_data.to(device)
    attentions   = []

    # Hook sur la couche d'auto-attention de la premiere couche du Transformer
    def _hook(module, inp, out):
        # MultiheadAttention : out = (attn_output, attn_weights) si need_weights=True
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            attentions.append(out[1].detach().cpu())

    layer0    = transformer.layers[0]
    self_attn = layer0.self_attn

    # TransformerEncoderLayer appelle self_attn avec need_weights=False.
    # On patche temporairement le forward de self_attn pour forcer need_weights=True,
    # puis on capture les poids via un hook.
    _orig_attn_fwd = self_attn.forward

    def _attn_fwd_patched(*args, **kwargs):
        kwargs["need_weights"] = True
        return _orig_attn_fwd(*args, **kwargs)

    self_attn.forward = _attn_fwd_patched
    hook_hdl = self_attn.register_forward_hook(_hook)

    # PyTorch >= 1.11 utilise un fast-path C++ en eval qui bypasse les hooks.
    # On force le mode train sur l'encodeur uniquement pour capturer les poids.
    was_training = enc.training
    enc.train()
    with torch.no_grad():
        enc(landsat_data)
    enc.train(was_training)

    hook_hdl.remove()
    self_attn.forward = _orig_attn_fwd   # restaure le forward original

    if not attentions:
        print("WARN : poids d'attention non captures meme avec need_weights=True")
        return None

    # (B, n_heads, n_years, n_years) → moyenne sur batch et tetes
    attn = attentions[0]   # (B, n_heads, T, T) ou (B, T, T)
    if attn.dim() == 4:
        attn_mean = attn.mean(dim=(0, 1)).numpy()   # (T, T)
    else:
        attn_mean = attn.mean(dim=0).numpy()

    T = attn_mean.shape[0]
    if year_labels is None:
        year_labels = [str(2000 + i) for i in range(T)]

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(attn_mean, cmap="viridis", aspect="auto")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(year_labels[:T], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(year_labels[:T], fontsize=7)
        ax.set_xlabel("Vers annee")
        ax.set_ylabel("Depuis annee")
        ax.set_title("Attention Transformer Landsat (moy. sur tetes et batch)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Attention Landsat -> {}".format(output_path))

    return attn_mean


# =============================================================================
# D. FINALISATION - noeud d'explication JSON par survey
# =============================================================================

def _extract_temporal_focus(attn_mean, year_labels):
    # type: (np.ndarray, List[str]) -> str
    """
    Retourne l'annee la plus 'regardee' par le Transformer Landsat.
    On utilise la somme des poids d'attention recus par chaque position (colonne).
    """
    col_sums = attn_mean.sum(axis=0)   # (T,) - importance de chaque annee
    best_t   = int(np.argmax(col_sums))
    return year_labels[best_t] if best_t < len(year_labels) else str(best_t)


def _compute_confidence(model, adapter, batch, device):
    # type: (nn.Module, ModelAdapter, object, str) -> str
    """
    Evalue la confiance du modele = probabilite moyenne des top-10 predictions.
    Retourne "High" / "Medium" / "Low".
    """
    try:
        with torch.no_grad():
            logits = adapter.forward(batch, model, device)
        probs   = torch.sigmoid(logits).cpu().numpy()
        top10   = float(np.sort(probs[0])[-10:].mean())
        if top10 >= 0.4:
            return "High"
        if top10 >= 0.2:
            return "Medium"
        return "Low"
    except Exception:
        return "Unknown"


def finalize_explanation_node(
    survey_id,          # identifiant du survey (int ou str)
    shap_results,       # dict retourne par shap_tabular()
    attn_mean,          # np.ndarray (T, T) retourne par landsat_attention()
    output_dir,         # dossier du batch courant (contient les .png)
    feature_names,      # liste des noms de features
    confidence="Unknown",
    year_labels=None,   # type: Optional[List[str]]
):
    # type: (...) -> str
    """
    Genere un fichier summary.json dans output_dir/survey_<survey_id>/.

    Contient :
      - les 5 features SHAP les plus influentes
      - l'annee Landsat la plus importante
      - la confiance du modele
      - les chemins vers les visuels

    Retourne le chemin du fichier JSON cree.
    """
    import json

    node_path = os.path.join(output_dir, "survey_{}".format(survey_id))
    os.makedirs(node_path, exist_ok=True)

    # 1. Top-5 drivers SHAP
    drivers = []
    if shap_results is not None:
        sv   = shap_results["shap_values"]       # (n_explain, n_feats)
        names = shap_results["feature_names"]
        mean_abs = np.abs(sv).mean(axis=0)       # (n_feats,)
        top_idx  = np.argsort(mean_abs)[-5:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            drivers.append({
                "rank":         rank,
                "feature":      names[idx],
                "mean_abs_shap": float(mean_abs[idx]),
            })

    # 2. Focus temporel depuis la matrice d'attention
    T = 21
    if year_labels is None:
        year_labels = [str(2000 + i) for i in range(T)]
    temporal_focus = (
        _extract_temporal_focus(attn_mean, year_labels)
        if attn_mean is not None else "N/A"
    )

    # 3. Chemins vers les visuels (relatifs au node_path)
    def _rel(fname):
        p = os.path.join(output_dir, fname)
        return os.path.relpath(p, node_path) if os.path.exists(p) else None

    summary = {
        "survey_id": survey_id,
        "visual_assets": {
            "gradcam":       _rel("gradcam.png"),
            "shap_plot":     os.path.relpath(
                                 os.path.join(os.path.dirname(output_dir), "shap", "shap_importance.png"),
                                 node_path,
                             ) if shap_results else None,
            "landsat_plot":  _rel("landsat_attention.png"),
        },
        "technical_summary": {
            "main_drivers":   drivers,
            "temporal_focus": temporal_focus,
            "model_confidence": confidence,
        },
    }

    json_path = os.path.join(node_path, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Node XAI -> {}".format(json_path))
    return json_path


# =============================================================================
# POINT D'ENTREE PRINCIPAL
# =============================================================================

def explain(
    model,                              # type: nn.Module
    adapter,                            # type: Union[str, ModelAdapter]
    val_loader,                         # iterable de batches
    device="cuda",                      # type: str
    output_dir="./explanations",        # type: str
    n_batches=1,                        # type: int
    species_idx=0,                      # type: int
    feature_names=None,                 # type: Optional[List[str]]
    run_gradcam=True,                   # type: bool
    run_shap=True,                      # type: bool
    run_attention=True,                 # type: bool
    run_finalize=True,                  # type: bool
    year_labels=None,                   # type: Optional[List[str]]
    **adapter_kwargs
):
    """
    Lance le framework d'explication complet sur un modele quelconque.

    Args:
        model        : modele PyTorch entraine
        adapter      : "malala", "adrian", "generic" ou instance ModelAdapter
        val_loader   : DataLoader de validation
        device       : "cuda" ou "cpu"
        output_dir   : dossier de sortie
        n_batches    : nombre de batches a analyser
        species_idx  : indice de l'espece cible pour GradCAM
        feature_names: noms des features tabulaires (optionnel)
        run_gradcam  : activer/desactiver GradCAM
        run_shap     : activer/desactiver SHAP
        run_attention: activer/desactiver l'analyse d'attention Landsat

    Returns:
        dict avec les resultats par batch
    """
    # Resoudre l'adaptateur si c'est un string
    if isinstance(adapter, str):
        adapter = get_adapter(adapter, **adapter_kwargs)

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to(device)

    # Collecte des batches
    batches = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        batches.append(batch)

    if not batches:
        print("ERREUR : val_loader est vide")
        return {}

    results = {}

    for i, batch in enumerate(batches):
        print("\n" + "=" * 60)
        print("Batch {}/{} - explications".format(i + 1, len(batches)))
        print("=" * 60)

        batch_dir = os.path.join(output_dir, "batch_{}".format(i))
        os.makedirs(batch_dir, exist_ok=True)
        batch_results = {"output_dir": batch_dir}

        # A. GradCAM
        if run_gradcam:
            path = os.path.join(batch_dir, "gradcam.png")
            attr = gradcam(model, adapter, batch,
                           species_idx=species_idx, device=device, output_path=path)
            batch_results["gradcam"] = attr

        # B. SHAP (utilise tous les batches collectes comme background)
        if run_shap and i == 0:   # une seule fois suffit
            shap_dir  = os.path.join(output_dir, "shap")
            shap_res  = shap_tabular(model, adapter, batches,
                                     feature_names=feature_names,
                                     n_background=min(50, len(batches) * 32),
                                     n_explain=min(20, len(batches) * 32),
                                     device=device, output_dir=shap_dir)
            results["shap"] = shap_res

        # C. Attention Landsat
        attn = None
        if run_attention:
            path = os.path.join(batch_dir, "landsat_attention.png")
            attn = landsat_attention(model, adapter, batch,
                                     output_path=path, device=device,
                                     year_labels=year_labels)
            batch_results["landsat_attention"] = attn

        # D. Noeud JSON final
        if run_finalize:
            shap_res   = results.get("shap")
            confidence = _compute_confidence(model, adapter, batch, device)
            survey_id  = i   # remplace par le vrai survey_id si disponible
            finalize_explanation_node(
                survey_id     = survey_id,
                shap_results  = shap_res,
                attn_mean     = attn,
                output_dir    = batch_dir,
                feature_names = feature_names or [],
                confidence    = confidence,
                year_labels   = year_labels,
            )

        results["batch_{}".format(i)] = batch_results

    print("\n" + "=" * 60)
    print("Termine. Fichiers dans : {}".format(output_dir))
    print("=" * 60)
    return results
