from __future__ import annotations

"""Compare real WITSML process evidence with the simulator's coarse states.

The comparison is intentionally limited to state occupancy and dwell-time
structure.  It does not compare absolute electrical power because the public
WITSML archive contains no total active-power channel.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
COARSE_STATES = ("drilling", "circulation", "hoisting", "idle_or_other", "unknown")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_coarse_state(values: pd.Series) -> pd.Series:
    return values.where(values.isin(COARSE_STATES), "idle_or_other")


def fractions_and_dwell(states: pd.Series, timestamps: pd.Series) -> tuple[dict, list[float]]:
    states = pd.Series(states).astype(str).reset_index(drop=True)
    timestamps = pd.to_datetime(
        timestamps, format="mixed", utc=True
    ).reset_index(drop=True)
    fractions = states.value_counts(normalize=True).reindex(COARSE_STATES, fill_value=0.0).to_dict()
    runs = states.ne(states.shift()).cumsum()
    grouped = pd.DataFrame({"state": states, "timestamp": timestamps, "run": runs}).groupby("run")
    dwell: list[float] = []
    for _, group in grouped:
        if len(group) < 1:
            continue
        duration = (group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]).total_seconds() / 60.0
        dwell.append(max(duration, 1.0))
    return {key: float(fractions[key]) for key in COARSE_STATES}, dwell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--simulation", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    external = pd.read_csv(args.external, parse_dates=["timestamp"])
    if "external_coarse_state_1min" not in external:
        raise RuntimeError("external file lacks one-minute state labels")
    external_fraction, external_dwell = fractions_and_dwell(
        external["external_coarse_state_1min"], external["timestamp"]
    )
    rows = []
    dwell_rows = []
    for source in (args.external, *args.simulation):
        if source == args.external:
            label = "real_witsml"
            fraction, dwell = external_fraction, external_dwell
            frame = external
        else:
            label = source.stem
            frame = pd.read_parquet(source)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame["coarse_state"] = synthetic_coarse_state(frame["operation_state"])
            # Use the same one-minute modal aggregation as the real archive.
            frame["minute"] = frame["timestamp"].dt.floor("min")
            minute = frame.groupby("minute", sort=True)["coarse_state"].agg(
            lambda values: values.mode().iat[0] if not values.mode().empty else "unknown"
            )
            fraction, dwell = fractions_and_dwell(
                minute.to_numpy(), minute.index.to_series()
            )
        for state in COARSE_STATES:
            rows.append({"source": label, "state": state, "fraction": fraction[state]})
        for value in dwell:
            dwell_rows.append({"source": label, "dwell_minutes": value})

    fractions = pd.DataFrame(rows)
    dwell_frame = pd.DataFrame(dwell_rows)
    fractions.to_csv(args.output_dir / "state_fraction_comparison.csv", index=False)
    dwell_frame.to_csv(args.output_dir / "state_dwell_comparison.csv", index=False)
    real = fractions.loc[fractions["source"] == "real_witsml"].set_index("state")["fraction"]
    sim = fractions.loc[fractions["source"] != "real_witsml"].groupby("state")["fraction"].mean()
    real_known_denominator = max(1.0 - float(real["unknown"]), 1e-12)
    sim_known_denominator = max(1.0 - float(sim["unknown"]), 1e-12)
    real_known = real / real_known_denominator
    sim_known = sim / sim_known_denominator
    summary = {
        "real_source_sha256": sha256(args.external),
        "simulation_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.simulation
        ],
        "real_state_fraction_all_rows": {state: float(real[state]) for state in COARSE_STATES},
        "simulation_mean_state_fraction_all_rows": {state: float(sim[state]) for state in COARSE_STATES},
        "real_state_fraction_known_rows": {state: float(real_known[state]) for state in COARSE_STATES if state != "unknown"},
        "simulation_mean_state_fraction_known_rows": {state: float(sim_known[state]) for state in COARSE_STATES if state != "unknown"},
        "real_known_state_coverage_fraction": float(1.0 - real["unknown"]),
        "simulation_known_state_coverage_fraction": float(1.0 - sim["unknown"]),
        "mean_absolute_fraction_difference_known_states": float(
            np.mean([abs(float(real_known[state]) - float(sim_known[state])) for state in COARSE_STATES if state != "unknown"])
        ),
        "real_dwell_p50_minutes": float(
            dwell_frame.loc[dwell_frame["source"] == "real_witsml", "dwell_minutes"].median()
        ),
        "real_dwell_p95_minutes": float(
            dwell_frame.loc[dwell_frame["source"] == "real_witsml", "dwell_minutes"].quantile(0.95)
        ),
        "simulation_dwell_p50_minutes": float(
            dwell_frame.loc[dwell_frame["source"] != "real_witsml", "dwell_minutes"].median()
        ),
        "simulation_dwell_p95_minutes": float(
            dwell_frame.loc[dwell_frame["source"] != "real_witsml", "dwell_minutes"].quantile(0.95)
        ),
        "interpretation": "external real-process plausibility check only; not total-active-power or site-level validation",
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
