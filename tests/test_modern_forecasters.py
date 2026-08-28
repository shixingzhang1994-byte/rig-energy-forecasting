from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.models import (  # noqa: E402
    ITransformerForecaster,
    PatchTSTForecaster,
    StateAwarePatchTransformer,
)


def test_modern_forecasters_produce_the_required_multi_step_shape():
    x = torch.randn(3, 120, 4)
    state = torch.zeros(3, 120, dtype=torch.long)
    patchtst = PatchTSTForecaster(
        input_size=4,
        history=120,
        horizon=12,
        patch_length=16,
        patch_stride=8,
        hidden_size=32,
        attention_heads=4,
        layers=2,
        dropout=0.0,
    )
    itransformer = ITransformerForecaster(
        input_size=4,
        history=120,
        horizon=12,
        hidden_size=32,
        attention_heads=4,
        layers=2,
        dropout=0.0,
    )
    assert patchtst(x, state).shape == (3, 12)
    assert itransformer(x, state).shape == (3, 12)


def test_patchtst_channel_independence_prevents_auxiliary_channel_leakage():
    torch.manual_seed(7)
    model = PatchTSTForecaster(
        input_size=4,
        history=120,
        horizon=12,
        patch_length=16,
        patch_stride=8,
        hidden_size=16,
        attention_heads=4,
        layers=1,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 120, 4)
    changed = x.clone()
    changed[:, :, 1:] = torch.randn_like(changed[:, :, 1:]) * 100.0
    with torch.inference_mode():
        original_output = model(x)
        changed_output = model(changed)
    assert torch.allclose(original_output, changed_output, atol=1e-6)


def test_instance_normalized_forecasters_remain_finite_on_constant_history():
    x = torch.ones(2, 120, 4)
    models = [
        PatchTSTForecaster(4, 120, 12, 16, 8, 16, 4, 1, 0.0),
        ITransformerForecaster(4, 120, 12, 16, 4, 1, 0.0),
    ]
    for model in models:
        assert torch.isfinite(model(x)).all()


def test_state_aware_patch_transformer_uses_operation_state_and_keeps_shape():
    torch.manual_seed(11)
    model = StateAwarePatchTransformer(
        numeric_input_size=4,
        num_states=6,
        state_embedding_dim=8,
        history=120,
        horizon=12,
        patch_length=16,
        patch_stride=8,
        hidden_size=32,
        attention_heads=4,
        layers=2,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 120, 4)
    idle = torch.zeros(2, 120, dtype=torch.long)
    drilling = torch.full((2, 120), 2, dtype=torch.long)
    with torch.inference_mode():
        idle_output = model(x, idle)
        drilling_output = model(x, drilling)
    assert idle_output.shape == (2, 12)
    assert not torch.allclose(idle_output, drilling_output)
