from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TemporalTransformerClassifier(nn.Module):
    """Small Transformer encoder for short-horizon risk classification."""

    def __init__(
        self,
        horizon: int,
        static_size: int,
        num_classes: int = 4,
        hidden_size: int = 64,
        attention_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.sequence_projection = nn.Linear(2, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, horizon, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.static_net = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        delta = torch.cat(
            [torch.zeros_like(sequence[:, :1]), sequence[:, 1:] - sequence[:, :-1]], dim=1
        )
        tokens = torch.stack([sequence, delta], dim=-1)
        encoded = self.encoder(self.sequence_projection(tokens) + self.position)
        temporal = encoded.mean(dim=1)
        context = self.static_net(static)
        return self.head(torch.cat([temporal, context], dim=1))


@dataclass
class TransformerTrainingResult:
    model: TemporalTransformerClassifier
    best_epoch: int
    validation_macro_f1: float
    history: list[dict[str, float]]


def _predict_probabilities(
    model: nn.Module,
    sequence: np.ndarray,
    static: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dataset = TensorDataset(
        torch.from_numpy(sequence.astype(np.float32)),
        torch.from_numpy(static.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for seq_batch, static_batch in loader:
            logits = model(seq_batch.to(device), static_batch.to(device))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_transformer_classifier(
    train_sequence: np.ndarray,
    train_static: np.ndarray,
    train_labels: np.ndarray,
    val_sequence: np.ndarray,
    val_static: np.ndarray,
    val_labels: np.ndarray,
    *,
    device: torch.device,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> TransformerTrainingResult:
    model = TemporalTransformerClassifier(
        horizon=train_sequence.shape[1],
        static_size=train_static.shape[1],
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        dropout=dropout,
    ).to(device)
    counts = np.bincount(train_labels, minlength=4).astype(float)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_sequence.astype(np.float32)),
        torch.from_numpy(train_static.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.int64)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -np.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for seq_batch, static_batch, label_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq_batch.to(device), static_batch.to(device))
            loss = criterion(logits, label_batch.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probability = _predict_probabilities(
            model, val_sequence, val_static, device, batch_size
        )
        val_f1 = float(f1_score(val_labels, val_probability.argmax(axis=1), average="macro"))
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_f1}
        )
        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return TransformerTrainingResult(model, best_epoch, float(best_f1), history)


def predict_transformer_probabilities(
    model: nn.Module,
    sequence: np.ndarray,
    static: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    return _predict_probabilities(model, sequence, static, device, batch_size)


def fit_probability_ensemble(
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    l2_penalty: float = 1e-4,
) -> dict[str, float]:
    names = list(probabilities)
    stack = np.stack([probabilities[name] for name in names], axis=0)
    label_index = np.arange(len(labels))
    class_counts = np.bincount(labels, minlength=stack.shape[-1]).astype(float)
    sample_weights = class_counts.sum() / np.maximum(class_counts[labels], 1.0)
    sample_weights /= sample_weights.mean()

    def objective(weights: np.ndarray) -> float:
        combined = np.tensordot(weights, stack, axes=(0, 0))
        selected = np.clip(combined[label_index, labels], 1e-8, 1.0)
        return float(
            -(sample_weights * np.log(selected)).mean()
            + l2_penalty * np.square(weights).sum()
        )

    initial = np.full(len(names), 1.0 / len(names))
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    weights = result.x if result.success else initial
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return {name: float(weight) for name, weight in zip(names, weights)}


def apply_probability_ensemble(
    probabilities: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    result = np.zeros_like(next(iter(probabilities.values())), dtype=float)
    for name, probability in probabilities.items():
        result += float(weights[name]) * probability
    result = np.maximum(result, 0.0)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)
