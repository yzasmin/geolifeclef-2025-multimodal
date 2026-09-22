from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchEncoder(nn.Module):
    def __init__(self, channels: list[int], token_dim: int) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.stem = ConvBlock(4, c1, stride=2)
        self.layer2 = ConvBlock(c1, c2, stride=2)
        self.layer3 = ConvBlock(c2, c3, stride=2)
        self.proj = nn.Conv2d(c3, token_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B, N, D
        pooled = tokens.mean(dim=1)
        return tokens, pooled


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(c_in, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                ]
            )
            c_in = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: B, C, T
        x = self.net(x)
        tokens = x.transpose(1, 2)  # B, T, D
        pooled = tokens.mean(dim=1)
        return tokens, pooled


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
