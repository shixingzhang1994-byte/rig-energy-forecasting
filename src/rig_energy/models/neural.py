from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        encoded, _ = self.lstm(x)
        return self.head(encoded[:, -1])


class Chomp1d(nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.amount] if self.amount > 0 else x


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        channels = input_size
        for level in range(levels):
            blocks.append(
                TemporalResidualBlock(
                    channels,
                    hidden_size,
                    kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            channels = hidden_size
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.transpose(1, 2)).transpose(1, 2)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        levels: int,
        kernel_size: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(input_size, hidden_size, levels, kernel_size, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.head(encoded[:, -1])


class StateAwareTCNAttention(nn.Module):
    """Project method: state embedding + causal TCN + attention + residual forecast.

    The model forecasts a correction relative to the most recent load, which
    stabilizes normal periods while allowing the state embedding and attention
    path to react to drilling-operation transitions.
    """

    def __init__(
        self,
        numeric_input_size: int,
        num_states: int,
        state_embedding_dim: int,
        hidden_size: int,
        levels: int,
        kernel_size: int,
        attention_heads: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.state_embedding = nn.Embedding(num_states, state_embedding_dim)
        self.input_projection = nn.Linear(numeric_input_size + state_embedding_dim, hidden_size)
        self.encoder = TemporalEncoder(hidden_size, hidden_size, levels, kernel_size, dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if state is None:
            raise ValueError("StateAwareTCNAttention 需要 operation_state_code")
        state_features = self.state_embedding(state)
        projected = self.input_projection(torch.cat([x, state_features], dim=-1))
        encoded = self.encoder(projected)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        attended = self.attention_norm(attended + encoded)
        weights = torch.softmax(self.pool_score(attended).squeeze(-1), dim=-1)
        pooled = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        context = torch.cat([attended[:, -1], pooled], dim=-1)
        residual = self.head(context)
        last_load = x[:, -1, 0].unsqueeze(-1)
        return last_load + residual


class PatchTSTForecaster(nn.Module):
    """PatchTST baseline with patching and channel-independent encoding.

    Each variate is encoded independently with shared Transformer weights.  The
    power-channel head therefore cannot borrow the categorical state pathway of
    the project model, keeping this a recognizable, auditable PatchTST baseline.
    """

    def __init__(
        self,
        input_size: int,
        history: int,
        horizon: int,
        patch_length: int,
        patch_stride: int,
        hidden_size: int,
        attention_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_length > history:
            raise ValueError("patch_length 不能大于历史窗口")
        self.input_size = int(input_size)
        self.history = int(history)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.patch_count = 1 + (history - patch_length) // patch_stride
        self.patch_projection = nn.Linear(patch_length, hidden_size)
        self.position = nn.Parameter(
            torch.zeros(1, self.patch_count, hidden_size)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(self.patch_count * hidden_size),
            nn.Linear(self.patch_count * hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1:] != (self.history, self.input_size):
            raise ValueError(
                f"PatchTST 期望 [B,{self.history},{self.input_size}]，实际 {tuple(x.shape)}"
            )
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / std
        # unfold: [B, patch_count, C, patch_length]
        patches = normalized.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 2, 1, 3)
        batch, channels, patch_count, patch_length = patches.shape
        tokens = self.patch_projection(
            patches.reshape(batch * channels, patch_count, patch_length)
        )
        encoded = self.encoder(tokens + self.position)
        power_encoded = encoded.reshape(batch, channels, patch_count, -1)[:, 0]
        normalized_prediction = self.head(power_encoded)
        return normalized_prediction * std[:, 0, 0].unsqueeze(-1) + mean[
            :, 0, 0
        ].unsqueeze(-1)


class ITransformerForecaster(nn.Module):
    """iTransformer baseline: whole variate histories are attention tokens."""

    def __init__(
        self,
        input_size: int,
        history: int,
        horizon: int,
        hidden_size: int,
        attention_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.history = int(history)
        self.series_projection = nn.Linear(history, hidden_size)
        self.variable_embedding = nn.Parameter(
            torch.zeros(1, input_size, hidden_size)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1:] != (self.history, self.input_size):
            raise ValueError(
                f"iTransformer 期望 [B,{self.history},{self.input_size}]，实际 {tuple(x.shape)}"
            )
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / std
        tokens = self.series_projection(normalized.transpose(1, 2))
        encoded = self.encoder(tokens + self.variable_embedding)
        normalized_prediction = self.head(encoded[:, 0])
        return normalized_prediction * std[:, 0, 0].unsqueeze(-1) + mean[
            :, 0, 0
        ].unsqueeze(-1)


class StateAwarePatchTransformer(nn.Module):
    """Project model: operation-state embeddings fused into temporal patches."""

    def __init__(
        self,
        numeric_input_size: int,
        num_states: int,
        state_embedding_dim: int,
        history: int,
        horizon: int,
        patch_length: int,
        patch_stride: int,
        hidden_size: int,
        attention_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_length > history:
            raise ValueError("patch_length 不能大于历史窗口")
        self.numeric_input_size = int(numeric_input_size)
        self.history = int(history)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.patch_count = 1 + (history - patch_length) // patch_stride
        self.state_embedding = nn.Embedding(num_states, state_embedding_dim)
        patch_input_size = patch_length * (numeric_input_size + state_embedding_dim)
        self.patch_projection = nn.Linear(patch_input_size, hidden_size)
        self.position = nn.Parameter(
            torch.zeros(1, self.patch_count, hidden_size)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(self.patch_count * hidden_size),
            nn.Linear(self.patch_count * hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if state is None:
            raise ValueError("StateAwarePatchTransformer 需要 operation_state_code")
        if x.shape[1:] != (self.history, self.numeric_input_size):
            raise ValueError(
                "StateAwarePatchTransformer 的输入形状与配置不一致"
            )
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / std
        fused = torch.cat([normalized, self.state_embedding(state)], dim=-1)
        # [B, patch_count, features, patch_length] -> flatten each temporal patch.
        patches = fused.unfold(1, self.patch_length, self.patch_stride)
        batch, patch_count, feature_count, patch_length = patches.shape
        tokens = self.patch_projection(
            patches.reshape(batch, patch_count, feature_count * patch_length)
        )
        encoded = self.encoder(tokens + self.position)
        residual = self.head(encoded)
        normalized_prediction = normalized[:, -1, 0].unsqueeze(-1) + residual
        return normalized_prediction * std[:, 0, 0].unsqueeze(-1) + mean[
            :, 0, 0
        ].unsqueeze(-1)
