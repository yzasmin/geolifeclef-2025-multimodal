from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, int] = (1024, 512), dropout: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout)))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_features, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dims: Tuple[int, int] = (256, 128), dropout: float = 0.1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FusionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fusion(x))


class MultimodalPlantModel(nn.Module):
    def __init__(
        self,
        n_species: int,
        env_dim: int,
        aux_dim: int,
        image_channels: int = 4,
        landsat_features: int = 24,
        climate_features: int = 76,
        branch_dim: int = 128,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.2,
        use_image_branch: bool = False,
    ) -> None:
        super().__init__()
        self.use_image_branch = use_image_branch
        self.image_encoder = ImageEncoder(in_channels=image_channels, embedding_dim=branch_dim) if use_image_branch else None
        self.landsat_encoder = TimeSeriesTransformerEncoder(landsat_features, embedding_dim=branch_dim, dropout=dropout)
        self.climate_encoder = TimeSeriesTransformerEncoder(climate_features, embedding_dim=branch_dim, dropout=dropout)
        self.environment_encoder = MLPEncoder(env_dim, embedding_dim=branch_dim, dropout=dropout)
        self.aux_encoder = MLPEncoder(aux_dim, embedding_dim=branch_dim, hidden_dims=(128, 128), dropout=dropout)
        self.head = FusionHead((4 + int(use_image_branch)) * branch_dim, n_species, hidden_dim=fusion_hidden_dim, dropout=dropout)

    def forward(
        self,
        env_features: torch.Tensor,
        aux_features: torch.Tensor,
        landsat_series: torch.Tensor,
        climate_series: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = [
            self.environment_encoder(env_features),
            self.aux_encoder(aux_features),
            self.landsat_encoder(landsat_series),
            self.climate_encoder(climate_series),
        ]
        if self.use_image_branch:
            if image is None:
                raise ValueError("Image branch is enabled but no image tensor was provided.")
            embeddings.insert(0, self.image_encoder(image))
        return self.head(torch.cat(embeddings, dim=1))
