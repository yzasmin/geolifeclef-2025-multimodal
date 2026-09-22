"""
Framework d'explication - Adrian/MultimodalPlantModel
======================================================

Usage :
    # Donnees factices (test rapide)
    python explain_example.py --gpu-id 0

    # Vraies donnees (val_loader reconstruit depuis le cache Adrian)
    python explain_example.py --gpu-id 0 --real-data

    # Options supplementaires
    python explain_example.py --gpu-id 0 --real-data --n-batches 3 --species-idx 42
    python explain_example.py --gpu-id 0 --real-data --no-shap   # plus rapide
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import numpy as np
import torch

# Rend Adrian/model.py et Adrian/train.py importables
_HERE      = os.path.dirname(os.path.abspath(__file__))   # Malala/
ADRIAN_DIR = os.path.join(_HERE, "..", "Adrian")
sys.path.insert(0, ADRIAN_DIR)

from explainability import explain

# ============================================================================
# Noms des features tabulaires (ameliore la lisibilite des graphiques SHAP)
# ============================================================================

# env_features = bioclim + elevation + human_footprint + landcover + soilgrids
# (64 features au total dans le checkpoint charge)
BIOCLIM_NAMES = [
    "bio01_AnnualMeanTemp", "bio02_MeanDiurnalRange", "bio03_Isothermality",
    "bio04_TempSeasonality", "bio05_MaxTempWarmest", "bio06_MinTempColdest",
    "bio07_TempAnnualRange", "bio08_MeanTempWettestQtr", "bio09_MeanTempDriestQtr",
    "bio10_MeanTempWarmestQtr", "bio11_MeanTempColdestQtr", "bio12_AnnualPrecip",
    "bio13_PrecipWettestMonth", "bio14_PrecipDriestMonth", "bio15_PrecipSeasonality",
    "bio16_PrecipWettestQtr", "bio17_PrecipDriestQtr", "bio18_PrecipWarmestQtr",
    "bio19_PrecipColdestQtr",
]
SOIL_NAMES = [
    "soil_sand", "soil_silt", "soil_clay", "soil_pH", "soil_OrgC",
    "soil_BulkDensity", "soil_CEC", "soil_CoarseFragments", "soil_TextureClass",
]
ELEV_NAMES    = ["elevation"]
FOOTPRINT_NAMES = ["footprint_{}".format(i) for i in range(22)]
LANDCOVER_NAMES = ["landcover"]

ENV_NAMES = BIOCLIM_NAMES + SOIL_NAMES + ELEV_NAMES + FOOTPRINT_NAMES + LANDCOVER_NAMES
# 19 + 9 + 1 + 22 + 1 = 52 - completer si env_dim > 52
# aux_features = lon, lat, year, geoUncertaintyInM, areaInM2 + one-hot region/country
AUX_NAMES_BASE = ["lon", "lat", "year", "geoUncertaintyInM", "areaInM2"]


# ============================================================================
# Chargement du checkpoint
# ============================================================================

def load_adrian_checkpoint(checkpoint_path, device):
    from model import MultimodalPlantModel

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint introuvable : {}".format(checkpoint_path))

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    env_dim          = state["environment_encoder.network.0.weight"].shape[1]
    aux_dim          = state["aux_encoder.network.0.weight"].shape[1]
    n_species        = state["head.classifier.bias"].shape[0]
    branch_dim       = state["landsat_encoder.input_projection.weight"].shape[0]
    use_image_branch = "image_encoder.encoder.0.block.0.weight" in state

    print("Checkpoint charge : {}".format(checkpoint_path))
    print("  env_dim={}, aux_dim={}, n_species={}, branch_dim={}, use_image={}".format(
        env_dim, aux_dim, n_species, branch_dim, use_image_branch))

    model = MultimodalPlantModel(
        n_species=n_species,
        env_dim=env_dim,
        aux_dim=aux_dim,
        landsat_features=24,
        climate_features=76,
        branch_dim=branch_dim,
        use_image_branch=use_image_branch,
    )
    model.load_state_dict(state)

    return model, {
        "env_dim": env_dim,
        "aux_dim": aux_dim,
        "n_species": n_species,
        "use_image_branch": use_image_branch,
    }


# ============================================================================
# DataLoader factice (test sans donnees)
# Format exact de collate_fn Adrian : (env, aux, landsat, climate, image, targets)
# ============================================================================

def make_dummy_loader(env_dim, aux_dim, n_species, use_image_branch,
                      n_samples=16, batch_size=8):
    class DummyDS(torch.utils.data.Dataset):
        def __len__(self):
            return n_samples

        def __getitem__(self, idx):
            image = torch.randn(4, 64, 64) if use_image_branch else None
            return (
                torch.randn(env_dim),                    # env
                torch.randn(aux_dim),                    # aux
                torch.randn(21, 24),                     # landsat  (21 ans, 24 feat)
                torch.randn(12, 76),                     # climate  (12 mois, 76 feat)
                image,                                   # image ou None
                (torch.rand(n_species) > 0.99).float(),  # label multi-hot
            )

    def collate(batch):
        env     = torch.stack([b[0] for b in batch])
        aux     = torch.stack([b[1] for b in batch])
        landsat = torch.stack([b[2] for b in batch])
        climate = torch.stack([b[3] for b in batch])
        image   = None if batch[0][4] is None else torch.stack([b[4] for b in batch])
        label   = torch.stack([b[5] for b in batch])
        return env, aux, landsat, climate, image, label

    return torch.utils.data.DataLoader(
        DummyDS(), batch_size=batch_size, shuffle=False, collate_fn=collate)


# ============================================================================
# DataLoader reel (depuis le cache prepare par Adrian/train.py)
# ============================================================================

def make_real_loader(use_image_branch, batch_size=8, fold_index=0,
                     n_folds=5, seed=42):
    """
    Reconstruit le val_loader d'Adrian a partir du cache PreparedData.
    Necessite que le cache (prepared_data.pkl) existe dans Adrian/artifacts/.
    """
    from train import (
        SurveyDataset, collate_fn,
        build_spatial_blocks, build_fold_masks,
        prepare_data,
    )
    from pathlib import Path

    # prepare_data() charge le cache s'il existe, le cree sinon.
    adrian     = Path(ADRIAN_DIR).resolve()
    # Cache dans Malala/ pour ne pas modifier le dossier d'Adrian
    cache_path = Path(_HERE) / "outputs" / "cache" / "prepared_multimodal.pkl"
    # Sinon, utilise le cache d'Adrian s'il existe deja
    adrian_cache = adrian / "artifacts" / "train_run" / "cache" / "prepared_multimodal.pkl"
    if adrian_cache.exists():
        cache_path = adrian_cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("Preparation des donnees (cache : {})...".format(cache_path))
    prepared = prepare_data(cache_path)

    n_classes = len(prepared.species_ids)
    groups    = build_spatial_blocks(prepared.metadata_train, grid_size=1.0)
    fold_masks = build_fold_masks(groups, n_folds=n_folds, seed=seed)

    val_mask = fold_masks[fold_index]
    val_idx  = np.where(val_mask)[0]

    val_dataset = SurveyDataset(
        prepared.env_train[val_mask],
        prepared.aux_train[val_mask],
        [prepared.landsat_train_paths[i] for i in val_idx],
        [prepared.climate_train_paths[i] for i in val_idx],
        [prepared.image_train_paths[i]   for i in val_idx],
        [prepared.train_labels[i]        for i in val_idx],
        n_classes,
        use_image_branch,
        prepared.landsat_mean,
        prepared.landsat_std,
        prepared.climate_mean,
        prepared.climate_std,
    )

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = functools.partial(collate_fn, n_classes=n_classes),
    )

    print("Val dataset : {} surveys, {} especes".format(len(val_dataset), n_classes))
    return loader, prepared.species_ids


# ============================================================================
# Noms des features
# ============================================================================

def build_feature_names(env_dim, aux_dim):
    env_names = list(ENV_NAMES[:env_dim])
    while len(env_names) < env_dim:
        env_names.append("env_{}".format(len(env_names)))

    aux_names = list(AUX_NAMES_BASE[:aux_dim])
    while len(aux_names) < aux_dim:
        aux_names.append("aux_{}".format(len(aux_names)))

    return env_names + aux_names


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="XAI - Adrian/MultimodalPlantModel")
    parser.add_argument("--checkpoint",   type=str, default=os.path.join(_HERE, "..", "Adrian", "artifacts", "train_run", "best_model.pt"))
    parser.add_argument("--gpu-id",       type=int, default=0)
    parser.add_argument("--output-dir",   type=str, default=os.path.join(_HERE, "explanations"))
    parser.add_argument("--n-batches",    type=int, default=1,  help="Nombre de batches a analyser")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--species-idx",  type=int, default=0,  help="Index espece cible pour GradCAM")
    parser.add_argument("--fold-index",   type=int, default=0,  help="Fold de validation (0-4)")
    parser.add_argument("--real-data",    action="store_true",  help="Utiliser les vraies donnees (cache Adrian)")
    parser.add_argument("--no-shap",      action="store_true",  help="Desactiver SHAP (plus rapide)")
    parser.add_argument("--no-attention", action="store_true",  help="Desactiver l'attention Landsat")
    args = parser.parse_args()

    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
    print("Device : {}".format(device))

    # 1. Modele
    model, meta = load_adrian_checkpoint(args.checkpoint, device)
    model.to(device).eval()

    env_dim          = meta["env_dim"]
    aux_dim          = meta["aux_dim"]
    n_species        = meta["n_species"]
    use_image_branch = meta["use_image_branch"]

    # 2. DataLoader
    if args.real_data:
        print("\nChargement des vraies donnees (cache PreparedData Adrian)...")
        val_loader, _ = make_real_loader(
            use_image_branch = use_image_branch,
            batch_size       = args.batch_size,
            fold_index       = args.fold_index,
        )
    else:
        print("\nCreation d'un DataLoader factice (utilise --real-data pour les vraies donnees)...")
        val_loader = make_dummy_loader(
            env_dim          = env_dim,
            aux_dim          = aux_dim,
            n_species        = n_species,
            use_image_branch = use_image_branch,
            n_samples        = (args.n_batches + 1) * args.batch_size,
            batch_size       = args.batch_size,
        )

    # 3. Noms des features
    feature_names = build_feature_names(env_dim, aux_dim)

    # 4. Framework XAI
    print("\n=== Framework XAI - Adrian/MultimodalPlantModel ===")
    results = explain(
        model         = model,
        adapter       = "adrian",
        val_loader    = val_loader,
        device        = device,
        output_dir    = args.output_dir,
        n_batches     = args.n_batches,
        species_idx   = args.species_idx,
        feature_names = feature_names,
        run_gradcam   = use_image_branch,
        run_shap      = not args.no_shap,
        run_attention = not args.no_attention,
    )

    # 5. Resume
    print("\nFichiers generes :")
    for info in results.values():
        if isinstance(info, dict) and "output_dir" in info:
            d = info["output_dir"]
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".png"):
                        print("  {}".format(os.path.join(d, f)))
    shap_dir = os.path.join(args.output_dir, "shap")
    if os.path.isdir(shap_dir):
        for f in os.listdir(shap_dir):
            if f.endswith(".png"):
                print("  {}".format(os.path.join(shap_dir, f)))


if __name__ == "__main__":
    main()
