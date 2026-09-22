# 🌱 Deep Plant Predict - Explainability Framework

## Structure du Framework

### 1. **GradCAM pour Images Sentinel-2** 🖼️
- Visualise quelles régions de l'image satellite influencent le modèle
- Output : heatmap superposée sur l'image RGB
- Cible : dernière couche convolutive (ResNet18)

### 2. **SHAP pour Données Tabulaires** 📊
- Explique l'importance de chaque feature (bioclim + env)
- Features SHAP : montre comment chaque variable affecte les prédictions
- Graphiques : feature importance + dependence plots
- Travaille sur les 19 features bioclim + 33 features env

### 3. **Attention Analysis pour Landsat** 📈
- Visualise les poids d'attention du Transformer
- Montre quelles années (sur 21) sont les plus pertinentes
- Heatmap : attention entre paires d'années

## Installation des dépendances

```bash
pip install shap captum torch torchvision matplotlib seaborn numpy
```

## Utilisation rapide

```python
from Malala.model import PlantModel, setup_device
from Malala.explainability import run_explainability_framework

device = setup_device(0)
model = PlantModel(use_image=True, use_landsat=True).to(device)
# model.load_state_dict(...)  # Chargez vos poids

run_explainability_framework(
    model=model,
    val_loader=your_val_loader,
    device=device,
    output_dir="./explanations",
    n_samples=5,
    species_idx=42,  # Espèce d'intérêt
)
```

## Outputs générés

```
./explanations/
├── sample_0/
│   ├── gradcam.png                    # GradCAM pour Sentinel-2
│   ├── shap_feature_importance.png   # Features les plus pertinentes
│   ├── shap_dependence.png           # Dépendance des top 4 features
│   └── landsat_attention.png         # Attention temporelle Landsat
├── sample_1/
│   └── ...
```

## Architecture du modèle (PlantModel)

| Branch | Input | Encoder | Output |
|--------|-------|---------|--------|
| **Sentinel-2** | (4, 64, 64) | ResNet18 + Projection | (256,) |
| **Landsat** | (6, 4, 21) | Transformer Encoder | (256,) |
| **Bioclim** | (19,) | MLP LayerNorm+GELU | (256,) |
| **Env** | (33,) | MLP LayerNorm+GELU | (256,) |
| **Fusion** | (1024,) | Linear + GELU + LayerNorm | (10k,) |

## Notes techniques

- **LayerNorm vs BatchNorm** : LayerNorm utilisé car plus stable pour données tabulaires (indépendant batch size)
- **GELU vs ReLU** : GELU apporte gradient non-nul pour x<0, meilleure régularisation
- **Transformer Landsat** : pre-norm (norm_first=True) pour gradients stables
- **ResNet18 pré-entraîné** : ImageNet weights → convergence 2-3x plus rapide

## Limitations et améliorations futures

- [ ] Intégrer MaskAttention pour une explication plus fine
- [ ] Supports multi-GPU pour plus de samples
- [ ] Export des attributions en format standardisé (JSON)
- [ ] Dashboard interactif avec Streamlit
