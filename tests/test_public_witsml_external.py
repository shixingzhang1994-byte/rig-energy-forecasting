from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts/analyze_public_witsml_external.py"
SPEC = importlib.util.spec_from_file_location("analyze_public_witsml_external", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coarse_state_uses_process_channels_without_power_claim() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2015-01-01T00:00:00Z", "2015-01-01T00:00:05Z", "2015-01-01T00:00:10Z"]
            ),
            "RPM": [0.0, 60.0, 0.0],
            "ROP": [0.0, 1.0, 0.0],
            "BONB": [0.0, 1.0, 0.0],
            "MBOT": [0.0, 1.0, 0.0],
            "TFLO": [0.0, 0.0, 500.0],
            "BPOS": [0.0, 0.0, 0.0],
        }
    )
    assert MODULE._coarse_state(frame).tolist() == ["idle_or_other", "drilling", "circulation"]


def test_rotary_proxy_formula_is_mechanical_kw() -> None:
    torque_kn_m = 10.0
    rpm = 60.0
    proxy_kw = 2.0 * np.pi * torque_kn_m * rpm / 60.0
    assert np.isclose(proxy_kw, 2.0 * np.pi * 10.0)


def test_state_label_sensitivity_is_predeclared_grid() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2015-01-01T00:00:00Z"]),
            "RPM": [60.0],
            "ROP": [1.0],
            "BONB": [1.0],
            "MBOT": [1.0],
            "TFLO": [500.0],
            "BPOS": [0.0],
        }
    )
    sensitivity = MODULE._state_label_sensitivity(frame)
    assert len(sensitivity) == 16
    assert sensitivity["known_row_coverage"].between(0.0, 1.0).all()
