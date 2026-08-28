from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.windowing import PowerScaler


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_neural_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    peak_weight: float = 0.0,
) -> tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x, state)
                raw_loss = torch.nn.functional.smooth_l1_loss(pred, y, reduction="none")
                if peak_weight > 0:
                    last_load = x[:, -1, 0].unsqueeze(-1)
                    weights = 1.0 + peak_weight * torch.clamp(torch.abs(y - last_load), 0.0, 2.0)
                    loss = torch.mean(raw_loss * weights)
                else:
                    loss = torch.mean(raw_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach()) * len(y)
            train_count += len(y)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                state = batch["state"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x, state)
                    loss = torch.nn.functional.smooth_l1_loss(pred, y)
                val_loss_sum += float(loss) * len(y)
                val_count += len(y)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    info = {
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model, info


def predict_neural_model(
    model: nn.Module,
    loader: DataLoader,
    scaler: PowerScaler,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    predictions = []
    targets = []
    transitions = []
    started = time.perf_counter()
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x, state)
            predictions.append(pred.float().cpu().numpy())
            targets.append(batch["y"].numpy())
            transitions.append(batch["transition"].numpy())
    elapsed = time.perf_counter() - started
    pred_scaled = np.concatenate(predictions)
    target_scaled = np.concatenate(targets)
    flags = np.concatenate(transitions).astype(bool)
    return scaler.inverse(target_scaled), scaler.inverse(pred_scaled), flags, elapsed


def save_torch_checkpoint(model: nn.Module, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)

