from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_v28_public_forecast_protocol_is_frozen_and_disjoint() -> None:
    config = yaml.safe_load(
        (PROJECT_DIR / "configs/v28_public_real_forecast_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    protocol = config["protocol"]
    assert protocol["status"] == "frozen_before_any_listed_test_target_was_read"
    assert set(config["datasets"]) == {
        "refit_house1",
        "refit_house2",
        "refit_house3",
        "refit_house4",
        "refit_house5",
        "refit_house11",
        "uci_household",
        "tsukuba_microgrid",
    }
    assert "opencem" not in config["datasets"]
    assert "opencem" in config["excluded_from_confirmation"]
    for dataset in config["datasets"].values():
        train_end = pd.Timestamp(dataset["train_target_end"])
        validation_start = pd.Timestamp(dataset["validation_target_start"])
        validation_end = pd.Timestamp(dataset["validation_target_end"])
        test_start = pd.Timestamp(dataset["test_target_start"])
        assert validation_start - train_end == pd.Timedelta(days=1, minutes=1)
        assert test_start - validation_end == pd.Timedelta(days=1, minutes=1)
        assert len(dataset["prepared_sha256"]) == 64


def test_v28_primary_family_and_model_selection_are_predeclared() -> None:
    config = yaml.safe_load(
        (PROJECT_DIR / "configs/v28_public_real_forecast_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    shared = config["shared_forecast"]
    assert config["protocol"]["primary_horizon_minutes"] == 1
    assert shared["candidate_models"] == ["ridge", "hist_gradient_boosting"]
    assert shared["selection_rule"] == (
        "lowest_validation_mae_then_refit_on_train_plus_validation"
    )
    assert shared["multiplicity"]["adjustment"] == "holm"
