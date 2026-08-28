from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PowerScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "PowerScaler":
        mean = float(np.mean(values))
        std = float(np.std(values))
        return cls(mean=mean, std=max(std, 1e-6))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def split_frame_by_time(
    frame: pd.DataFrame, train_fraction: float, val_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_stop = int(n * train_fraction)
    val_stop = int(n * (train_fraction + val_fraction))
    if train_stop <= 0 or val_stop <= train_stop or val_stop >= n:
        raise ValueError("数据切分比例或数据量不合理")
    return (
        frame.iloc[:train_stop].reset_index(drop=True),
        frame.iloc[train_stop:val_stop].reset_index(drop=True),
        frame.iloc[val_stop:].reset_index(drop=True),
    )


def _numeric_features(frame: pd.DataFrame, scaler: PowerScaler) -> np.ndarray:
    power = frame["total_active_power_kw"].to_numpy(dtype=np.float32)
    power_z = scaler.transform(power).astype(np.float32)
    delta_z = np.r_[0.0, np.diff(power) / scaler.std].astype(np.float32)
    timestamp = pd.to_datetime(frame["timestamp"])
    seconds_of_day = (
        timestamp.dt.hour.to_numpy() * 3600
        + timestamp.dt.minute.to_numpy() * 60
        + timestamp.dt.second.to_numpy()
    )
    phase = 2 * np.pi * seconds_of_day / 86400.0
    return np.column_stack([power_z, delta_z, np.sin(phase), np.cos(phase)]).astype(np.float32)


class WindowedRigDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        scaler: PowerScaler,
        history_steps: int,
        horizon_steps: int,
        stride: int,
    ) -> None:
        self.features = _numeric_features(frame, scaler)
        self.states = frame["operation_state_code"].to_numpy(dtype=np.int64)
        power = frame["total_active_power_kw"].to_numpy(dtype=np.float32)
        self.targets = scaler.transform(power).astype(np.float32)
        self.history = int(history_steps)
        self.horizon = int(horizon_steps)
        self.indices = np.arange(
            self.history,
            len(frame) - self.horizon + 1,
            max(1, int(stride)),
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        stop = int(self.indices[item])
        start = stop - self.history
        target_stop = stop + self.horizon
        x = torch.from_numpy(self.features[start:stop])
        state = torch.from_numpy(self.states[start:stop])
        y = torch.from_numpy(self.targets[stop:target_stop])
        transition = bool(np.any(self.states[start + 1 : target_stop] != self.states[start: target_stop - 1]))
        return {
            "x": x,
            "state": state,
            "y": y,
            "transition": torch.tensor(transition),
        }


def make_tree_windows(
    frame: pd.DataFrame,
    history_steps: int,
    horizon_steps: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = frame["total_active_power_kw"].to_numpy(dtype=np.float32)
    states = frame["operation_state_code"].to_numpy(dtype=np.int16)
    timestamp = pd.to_datetime(frame["timestamp"])
    seconds = (
        timestamp.dt.hour.to_numpy() * 3600
        + timestamp.dt.minute.to_numpy() * 60
        + timestamp.dt.second.to_numpy()
    )
    lag_candidates = [1, 2, 3, 6, 12, 24, 60, history_steps]
    lags = sorted({min(history_steps, lag) for lag in lag_candidates})
    indices = np.arange(history_steps, len(frame) - horizon_steps + 1, max(1, stride))
    features: list[list[float]] = []
    targets: list[np.ndarray] = []
    transitions: list[bool] = []
    for stop in indices:
        hist = power[stop - history_steps : stop]
        row = [float(power[stop - lag]) for lag in lags]
        for width in (6, 12, min(60, history_steps), history_steps):
            values = hist[-width:]
            row.extend([float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values))])
        phase = 2 * np.pi * seconds[stop - 1] / 86400.0
        row.extend([float(np.sin(phase)), float(np.cos(phase)), float(states[stop - 1])])
        features.append(row)
        targets.append(power[stop : stop + horizon_steps])
        transitions.append(bool(np.any(states[stop - history_steps + 1 : stop + horizon_steps] != states[stop - history_steps : stop + horizon_steps - 1])))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(transitions, dtype=bool),
    )

