from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.training import weighted_forecast_loss  # noqa: E402


def test_transition_weight_only_increases_transition_sample_contribution():
    pred = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    target = torch.zeros_like(pred)
    last_load = torch.zeros(2, 1)
    transition = torch.tensor([False, True])

    base = weighted_forecast_loss(pred, target, last_load, transition)
    emphasized = weighted_forecast_loss(
        pred,
        target,
        last_load,
        transition,
        transition_weight=1.0,
    )

    assert torch.isclose(base, torch.tensor(0.5))
    assert torch.isclose(emphasized, torch.tensor(0.75))


def test_peak_and_transition_weights_compose_multiplicatively():
    pred = torch.tensor([[2.0]])
    target = torch.tensor([[0.0]])
    last_load = torch.tensor([[2.0]])
    transition = torch.tensor([True])
    loss = weighted_forecast_loss(
        pred,
        target,
        last_load,
        transition,
        peak_weight=0.5,
        transition_weight=1.0,
    )
    # smooth_l1(2)=1.5, peak factor=2, transition factor=2.
    assert torch.isclose(loss, torch.tensor(6.0))
