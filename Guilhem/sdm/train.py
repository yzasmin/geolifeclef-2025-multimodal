from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sdm.losses import MaskedAsymmetricLoss, total_loss
from sdm.metrics import micro_precision_recall, sample_f1_score, topk_predictions
from sdm.model.hca_sdm import HCASDM


@dataclass
class TrainResult:
    best_state_dict: dict[str, Any]
    best_metrics: dict[str, float]


def build_model(
    num_species: int,
    landsat_channels: int,
    bioclim_channels: int,
    tabular_dim: int,
    model_cfg: dict,
    device: str,
) -> HCASDM:
    model = HCASDM(
        num_species=num_species,
        landsat_channels=landsat_channels,
        bioclim_channels=bioclim_channels,
        tabular_dim=tabular_dim,
        model_cfg=model_cfg,
    )
    return model.to(device)


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def train_model(
    model: HCASDM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_cfg: dict,
    min_k: int,
    max_k: int,
    device: str,
) -> TrainResult:
    epochs = int(training_cfg["epochs"])
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg["weight_decay"])
    amp = bool(training_cfg["amp"]) and device.startswith("cuda")
    patience = int(training_cfg["patience"])
    grad_clip = float(training_cfg["grad_clip"])
    log_interval = int(training_cfg.get("log_interval", 50))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    asl = MaskedAsymmetricLoss(
        gamma_pos=float(training_cfg["gamma_pos"]),
        gamma_neg=float(training_cfg["gamma_neg"]),
        clip=float(training_cfg["loss_clip"]),
    )
    set_size_weight = float(training_cfg["set_size_weight"])

    best_f1 = -1.0
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_metrics: dict[str, float] = {}
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}", leave=False), start=1):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(
                    patch=batch["patch"],
                    landsat=batch["landsat"],
                    bioclim_ts=batch["bioclim_ts"],
                    tabular=batch["tabular"],
                )
                labels = torch.clamp(batch["labels"], min=0.0)
                loss, loss_parts = total_loss(
                    logits=outputs["logits"],
                    targets=labels,
                    mask=batch["label_mask"],
                    set_size_pred=outputs["set_size"],
                    set_size_target=batch["set_size"],
                    asl_loss=asl,
                    set_size_weight=set_size_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
            if step % log_interval == 0:
                pass

        val_metrics = evaluate_pa(model, val_loader, min_k=min_k, max_k=max_k, device=device)
        epoch_loss = running_loss / max(1, len(train_loader))
        val_metrics["train_loss"] = epoch_loss

        if val_metrics["f1_samples"] > best_f1:
            best_f1 = val_metrics["f1_samples"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return TrainResult(best_state_dict=best_state, best_metrics=best_metrics)


@torch.no_grad()
def evaluate_pa(
    model: HCASDM,
    dataloader: DataLoader,
    min_k: int,
    max_k: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    y_true_all = []
    y_pred_all = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = _move_batch(batch, device)
        outputs = model(
            patch=batch["patch"],
            landsat=batch["landsat"],
            bioclim_ts=batch["bioclim_ts"],
            tabular=batch["tabular"],
        )

        probs = torch.sigmoid(outputs["logits"])
        pred = topk_predictions(probs, outputs["set_size"], min_k=min_k, max_k=max_k)

        y_true_all.append(batch["labels"].detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    f1_samples = sample_f1_score(y_true, y_pred)
    precision, recall = micro_precision_recall(y_true, y_pred)

    return {
        "f1_samples": float(f1_samples),
        "precision_micro": float(precision),
        "recall_micro": float(recall),
    }


@torch.no_grad()
def predict_scores(
    model: HCASDM,
    dataloader: DataLoader,
    device: str,
) -> dict[str, Any]:
    model.eval()
    survey_ids: list[int] = []
    probs_list = []
    set_size_list = []

    for batch in tqdm(dataloader, desc="predict", leave=False):
        survey_ids.extend([int(x) for x in batch["survey_id"]])
        batch = _move_batch(batch, device)
        outputs = model(
            patch=batch["patch"],
            landsat=batch["landsat"],
            bioclim_ts=batch["bioclim_ts"],
            tabular=batch["tabular"],
        )
        probs_list.append(torch.sigmoid(outputs["logits"]).detach().cpu())
        set_size_list.append(outputs["set_size"].detach().cpu())

    probs = torch.cat(probs_list, dim=0)
    set_sizes = torch.cat(set_size_list, dim=0)
    return {
        "survey_ids": survey_ids,
        "probs": probs,
        "set_sizes": set_sizes,
    }
