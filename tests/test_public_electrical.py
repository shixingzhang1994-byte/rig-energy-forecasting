from __future__ import annotations

from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_electrical import (
    prepare_m5bat_one_minute,
    prepare_opencem_one_minute,
    prepare_refit_house_one_minute,
    prepare_tsukuba_microgrid_one_minute,
    prepare_uci_household_one_minute,
)


def _row(ts: int, inverter: int, load_w: float, *, full: bool = True) -> dict[str, float]:
    return {
        "read_ts": ts,
        "inverter": inverter,
        "outsumw": load_w,
        "pv1power": 100.0 if full else np.nan,
        "battvolt": 52.0,
        "battcurr": 2.0,
        "battsoc": 80.0,
        "battchgpower": 50.0,
        "gridpowerw_a": 20.0,
        "linepowerw_a": 30.0 if full else np.nan,
    }


def test_opencem_adapter_prefers_complete_duplicate_and_sums_inverters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements.csv"
    rows: list[dict[str, float]] = []
    base = 1_700_000_000
    for offset in (0, 12, 24):
        rows.append(_row(base + offset, 1, 1_000.0))
        rows.append(_row(base + offset, 2, 500.0))
    # A shifted/incomplete duplicate has a misleading low load. The fixed
    # completeness rule must retain the physically complete record.
    rows.append(_row(base, 1, 5.0, full=False))
    pd.DataFrame(rows).to_csv(path, index=False)

    canonical, metadata = prepare_opencem_one_minute([path])

    assert len(canonical) == 1
    assert canonical.loc[0, "quality_ok"]
    assert np.isclose(canonical.loc[0, "measured_load_kw"], 1.5)
    assert metadata["conflicting_duplicate_keys"] == 1
    assert metadata["target_imputed"] is False
    assert metadata["supports_target_rig_field_validation"] is False


def test_opencem_adapter_marks_sparse_minutes_without_target_imputation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements.csv"
    base = 1_700_000_000
    rows = [
        _row(base, 1, 1_000.0),
        _row(base, 2, 500.0),
        _row(base + 120, 1, 1_100.0),
        _row(base + 120, 2, 600.0),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

    canonical, metadata = prepare_opencem_one_minute([path])

    assert len(canonical) == 3
    assert not canonical["quality_ok"].any()
    assert canonical.loc[1, "quality_issue"] == "insufficient_samples"
    assert np.isnan(canonical.loc[1, "measured_load_kw"])
    assert metadata["target_imputed"] is False


def test_opencem_adapter_rejects_register_wrap_current(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    base = 1_700_000_000
    rows = []
    for offset in (0, 12, 24):
        for inverter in (1, 2):
            row = _row(base + offset, inverter, 500.0)
            row["battcurr"] = 6_553.5
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

    _, metadata = prepare_opencem_one_minute([path])

    assert metadata["invalid_values_replaced_with_null"]["battcurr"] == 6


def test_refit_adapter_excludes_issue_rows_and_does_not_impute(tmp_path: Path) -> None:
    path = tmp_path / "house.csv"
    base = 1_700_000_000
    pd.DataFrame(
        {
            "Unix": [base, base + 10, base + 20, base + 120, base + 130],
            "Aggregate": [1000, 1100, 900, 1200, 1300],
            "Issues": [0, 0, 1, 0, 0],
        }
    ).to_csv(path, index=False)

    canonical, metadata = prepare_refit_house_one_minute(
        path,
        house_id="house_test",
        minimum_valid_samples_per_minute=2,
        minimum_issue_free_fraction=0.60,
    )

    assert canonical.loc[0, "quality_ok"]
    assert np.isclose(canonical.loc[0, "measured_load_kw"], 1.05)
    assert np.isnan(canonical.loc[1, "measured_load_kw"])
    assert canonical.loc[1, "quality_issue"] == "insufficient_valid_samples"
    assert metadata["issue_flagged_rows"] == 1
    assert metadata["target_imputed"] is False


def test_uci_adapter_keeps_missing_minute_unimputed(tmp_path: Path) -> None:
    path = tmp_path / "household.txt"
    path.write_text(
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity\n"
        "16/12/2006;17:24:00;4.216;0.418;234.840;18.400\n"
        "16/12/2006;17:25:00;?;?;?;?\n"
        "16/12/2006;17:26:00;5.360;0.436;233.630;23.000\n",
        encoding="utf-8",
    )

    canonical, metadata = prepare_uci_household_one_minute(path)

    assert canonical["quality_ok"].tolist() == [True, False, True]
    assert np.isnan(canonical.loc[1, "measured_load_kw"])
    assert metadata["quality_ok_fraction"] == 2 / 3
    assert metadata["target_imputed"] is False


def test_tsukuba_adapter_derives_balance_and_rejects_invalid_soc(tmp_path: Path) -> None:
    archive = tmp_path / "cleaned.zip"
    rows = [
        "#,Japanese labels",
        "#,10101,10105,10106,10201,10203,10307,12144,12152,20106,20109,20112,20115",
        "#,kW,V,A,V,kW,kW,kW,%,kW,kW,kW,kW",
    ]
    for second in range(4):
        rows.append(
            f"'2015/01/01 00:00:0{second},10,350,1,6600,500,50,10,"
            f"{80 if second < 3 else 180},10,10,10,20"
        )
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr("20150101SecCsv.csv", "\n".join(rows).encode("cp932"))

    canonical, metadata = prepare_tsukuba_microgrid_one_minute(
        archive,
        minimum_valid_samples_per_minute=3,
        minimum_valid_fraction=0.75,
        chunksize=2,
    )

    assert len(canonical) == 1
    assert canonical.loc[0, "quality_ok"]
    assert np.isclose(canonical.loc[0, "measured_load_kw"], 560.0)
    assert metadata["invalid_values_replaced_with_null"]["battery_soc_pct"] == 1
    assert metadata["target_imputed"] is False
    assert metadata["measured_total_active_power"] is False


def test_m5bat_adapter_joins_bms_and_bsc_without_imputation(tmp_path: Path) -> None:
    timestamps = pd.date_range("2020-01-01", periods=4, freq="s", tz="UTC")
    bms = pd.DataFrame(
        {
            "power_dc_W_bms": [1.0] * 4,
            "current_A_bms": [2.0] * 4,
            "voltage_bat_V_bms": [620.0] * 4,
            "soc_pct_bms": [80.0] * 4,
            "flag_imbalance_voltage_bms": [0] * 4,
            "flag_alarm_voltage_imbalance_bms": [0] * 4,
        },
        index=timestamps,
    )
    bms.index.name = "timestamp_utc"
    bsc = pd.DataFrame(
        {
            "power_ac_kW_bsc": [0.9] * 4,
            "frequency_Hz_bsc": [50.0] * 4,
            "temperature_degC_bsc": [25.0] * 4,
            "power_ac_setpoint_kW_bsc": [1.0] * 4,
            "flag_fcr_active_bsc": [1] * 4,
            "flag_standby_bsc": [0] * 4,
            "flag_stop_bsc": [0] * 4,
            "flag_failure_bsc": [0] * 4,
            "flag_const_bsc": [0] * 4,
        },
        index=timestamps,
    )
    bsc.index.name = "timestamp_utc"
    for year in range(2017, 2026):
        bms.to_parquet(tmp_path / f"Exide1_BMS_{year}.parquet")
        bsc.to_parquet(tmp_path / f"Exide1_BSC_{year}.parquet")
    (tmp_path / "codebook.csv").write_text("table,column_name\n", encoding="utf-8")

    canonical, metadata = prepare_m5bat_one_minute(
        tmp_path, minimum_samples_per_minute=3, batch_size=2
    )

    assert len(canonical) == 1
    assert canonical.loc[0, "quality_ok"]
    assert np.isclose(canonical.loc[0, "measured_battery_dc_power_kw"], 1.0)
    assert np.isclose(canonical.loc[0, "measured_ac_power_kw"], 0.9)
    assert metadata["target_imputed"] is False
    assert metadata["supports_real_bess_validation"] is True
