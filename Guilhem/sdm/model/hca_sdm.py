from __future__ import annotations

import torch
from torch import nn

from sdm.model.backbones import PatchEncoder, TabularEncoder, TemporalEncoder


class HCASDM(nn.Module):
    def __init__(
        self,
        num_species: int,
        landsat_channels: int,
        bioclim_channels: int,
        tabular_dim: int,
        model_cfg: dict,
    ) -> None:
        super().__init__()
        embed_dim = int(model_cfg["embed_dim"])
        attn_heads = int(model_cfg["attn_heads"])
        dropout = float(model_cfg["dropout"])

        patch_cfg = model_cfg["patch_encoder"]
        temporal_cfg = model_cfg["temporal_encoder"]
        tabular_cfg = model_cfg["tabular_encoder"]

        self.patch_encoder = PatchEncoder(
            channels=list(patch_cfg["channels"]),
            token_dim=int(patch_cfg["token_dim"]),
        )
        self.spatial_proj = nn.Linear(int(patch_cfg["token_dim"]), embed_dim)

        temporal_hidden = int(temporal_cfg["hidden_dim"])
        kernel_size = int(temporal_cfg["kernel_size"])
        num_layers = int(temporal_cfg["num_layers"])

        self.landsat_encoder = TemporalEncoder(
            in_channels=max(1, landsat_channels),
            hidden_dim=temporal_hidden,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )
        self.bioclim_encoder = TemporalEncoder(
            in_channels=max(1, bioclim_channels),
            hidden_dim=temporal_hidden,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )
        self.temporal_proj = nn.Linear(temporal_hidden, embed_dim)

        self.tabular_encoder = TabularEncoder(
            in_dim=max(1, tabular_dim),
            hidden_dims=list(tabular_cfg["hidden_dims"]),
            dropout=float(tabular_cfg["dropout"]),
        )
        self.context_proj = nn.Linear(int(tabular_cfg["hidden_dims"][-1]), embed_dim)

        self.cross_attn_context_temporal = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_eco_spatial = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.species_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_species),
        )
        self.set_size_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        patch: torch.Tensor,
        landsat: torch.Tensor,
        bioclim_ts: torch.Tensor,
        tabular: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spatial_tokens, spatial_pooled = self.patch_encoder(patch)
        spatial_tokens = self.spatial_proj(spatial_tokens)
        spatial_pooled = self.spatial_proj(spatial_pooled)

        landsat_tokens, landsat_pooled = self.landsat_encoder(landsat)
        bioclim_tokens, bioclim_pooled = self.bioclim_encoder(bioclim_ts)

        temporal_tokens = torch.cat([landsat_tokens, bioclim_tokens], dim=1)
        temporal_tokens = self.temporal_proj(temporal_tokens)

        temporal_pooled = self.temporal_proj((landsat_pooled + bioclim_pooled) / 2.0)

        context_vec = self.tabular_encoder(tabular)
        context_token = self.context_proj(context_vec).unsqueeze(1)

        eco_token, _ = self.cross_attn_context_temporal(
            query=context_token,
            key=temporal_tokens,
            value=temporal_tokens,
        )

        fused_token, _ = self.cross_attn_eco_spatial(
            query=eco_token,
            key=spatial_tokens,
            value=spatial_tokens,
        )

        fused = fused_token.squeeze(1)
        fused = fused + spatial_pooled + temporal_pooled + context_token.squeeze(1)
        fused = self.norm(self.dropout(fused))

        logits = self.species_head(fused)
        set_size = torch.nn.functional.softplus(self.set_size_head(fused)).squeeze(-1) + 1.0

        return {
            "logits": logits,
            "set_size": set_size,
        }
