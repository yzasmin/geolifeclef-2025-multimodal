from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Variantes testees puis retirees (gardees ici en memoire uniquement)
# ---------------------------------------------------------------------------
# - Swin-Tiny comme backbone image Sentinel
# - EfficientNet-B0 comme backbone image Sentinel
# Les variantes ci-dessus restent retirees. En revanche, la variante
# ConvNeXt-Tiny + attention pooling spatial est reactivee plus bas car c'est
# celle qui avait donne ~0.37 en validation.


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


class ImageEncoderSimple(nn.Module):
    """Original simple CNN encoder (kept for backward compat with old checkpoints)."""
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


class SpatialAttentionPooling(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        logits = self.attention(x).view(batch_size, -1)
        weights = torch.softmax(logits, dim=1).view(batch_size, 1, height, width)
        return (x * weights).sum(dim=(2, 3))


class ImageEncoder(nn.Module):
    """EfficientNet-B3 pretrained on ImageNet, modified for 4 channels (RGB+NIR).

    - Input conv modified: 3ch -> 4ch (NIR initialized from Red channel weights)
    - Features extracted before global pooling -> 1536-dim
    - Projected to embedding_dim via Linear
    - Gradient checkpointing enabled to save GPU memory
    """
    def __init__(self, in_channels: int = 4, embedding_dim: int = 128) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        # Load pretrained EfficientNet-B3
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Modify first conv: 3 channels -> 4 channels
        old_conv = base.features[0][0]  # Conv2d(3, 40, 3, stride=2, padding=1)
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            # Copy RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialize NIR channel by cloning Red channel weights
            if in_channels > 3:
                new_conv.weight[:, 3:, :, :] = old_conv.weight[:, 0:1, :, :]
        base.features[0][0] = new_conv

        # Keep only the feature extractor (no classifier)
        self.features = base.features       # -> (B, 1536, H, W)
        self.avgpool = base.avgpool          # AdaptiveAvgPool2d(1)
        self.projector = nn.Linear(1536, embedding_dim)

        # Enable gradient checkpointing on feature blocks
        self.use_checkpointing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            # Checkpoint through feature blocks to save memory
            x = checkpoint(self._forward_features, x, use_reentrant=False)
        else:
            x = self._forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 1536)
        return self.projector(x)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ImageEncoderConvNeXt(nn.Module):
    """ConvNeXt-Tiny pretrained on ImageNet, adapted to 4-channel Sentinel input."""
    def __init__(self, in_channels: int = 4, embedding_dim: int = 128) -> None:
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if in_channels > 3:
                new_conv.weight[:, 3:, :, :] = old_conv.weight[:, 0:1, :, :]
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        base.features[0][0] = new_conv

        self.features = base.features
        self.avgpool = base.avgpool
        out_dim = int(base.classifier[-1].in_features)
        self.attention_pool = SpatialAttentionPooling(out_dim)
        self.projector = nn.Linear(out_dim * 2, embedding_dim)
        self.use_checkpointing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            x = checkpoint(self._forward_features, x, use_reentrant=False)
        else:
            x = self._forward_features(x)
        avg_features = torch.flatten(self.avgpool(x), 1)
        attn_features = self.attention_pool(x)
        return self.projector(torch.cat([avg_features, attn_features], dim=1))

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ImageEncoderResNet50(nn.Module):
    """ResNet-50 pretrained on ImageNet, adapted to 4-channel Sentinel input."""
    def __init__(self, in_channels: int = 4, embedding_dim: int = 128) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if in_channels > 3:
                new_conv.weight[:, 3:, :, :] = old_conv.weight[:, 0:1, :, :]
        base.conv1 = new_conv

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.projector = nn.Linear(base.fc.in_features, embedding_dim)
        self.use_checkpointing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            x = checkpoint(self._forward_features, x, use_reentrant=False)
        else:
            x = self._forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projector(x)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


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
        image_backbone: str = "efficientnet_b3",
    ) -> None:
        super().__init__()
        self.use_image_branch = use_image_branch
        self.image_backbone = image_backbone
        if use_image_branch:
            if image_backbone == "convnext_tiny":
                self.image_encoder = ImageEncoderConvNeXt(in_channels=image_channels, embedding_dim=branch_dim)
            elif image_backbone == "resnet50":
                self.image_encoder = ImageEncoderResNet50(in_channels=image_channels, embedding_dim=branch_dim)
            else:
                self.image_encoder = ImageEncoder(in_channels=image_channels, embedding_dim=branch_dim)
        else:
            self.image_encoder = None
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
