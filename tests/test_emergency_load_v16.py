from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.safety.emergency_load import (  # noqa: E402
    allocate_priority_greedy,
    allocate_priority_highs,
    allocate_proportional,
)


def test_priority_allocation_sheds_lower_tiers_before_critical_load():
    demand = np.array([[400.0, 300.0, 300.0], [400.0, 300.0, 300.0]])
    available = np.array([800.0, 250.0])
    priority = allocate_priority_greedy(demand, available)
    assert np.allclose(priority.served_kw[0], [400.0, 300.0, 100.0])
    assert np.allclose(priority.served_kw[1], [250.0, 0.0, 0.0])
    proportional = allocate_proportional(demand, available)
    assert proportional.shed_kw[0, 0] > 0.0
    assert priority.shed_kw[0, 0] == 0.0


def test_highs_reference_matches_declared_lexicographic_policy():
    demand = np.array(
        [[500.0, 250.0, 250.0], [300.0, 300.0, 400.0], [100.0, 100.0, 100.0]]
    )
    available = np.array([700.0, 200.0, 500.0])
    greedy = allocate_priority_greedy(demand, available)
    highs = allocate_priority_highs(demand, available)
    assert np.allclose(highs.served_kw, greedy.served_kw, atol=1e-7)
    assert highs.solver_metadata["success"] is True
    assert highs.solver_metadata["highs_version"] != "embedded-version-unavailable"

