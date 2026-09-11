from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import STATE_TO_CODE


TRANSITIONS = {
    "idle": (("circulation", 0.45), ("drilling", 0.10), ("tripping", 0.25), ("maintenance", 0.20)),
    "circulation": (("drilling", 0.65), ("idle", 0.15), ("tripping", 0.10), ("maintenance", 0.10)),
    "drilling": (("connection", 0.62), ("circulation", 0.23), ("tripping", 0.07), ("idle", 0.08)),
    "connection": (("drilling", 0.82), ("circulation", 0.12), ("tripping", 0.02), ("idle", 0.04)),
    "tripping": (("circulation", 0.50), ("idle", 0.30), ("maintenance", 0.20)),
    "maintenance": (("idle", 0.70), ("circulation", 0.30)),
}


@dataclass(frozen=True)
class Segment:
    state: str
    start: int
    stop: int


def _colored_noise(rng: np.random.Generator, n: int, sigma: float, phi: float) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    scale = max(np.std(out), 1e-9)
    return out * (sigma / scale)


def _choose_next_state(rng: np.random.Generator, current: str) -> str:
    candidates = TRANSITIONS[current]
    names = [item[0] for item in candidates]
    probs = np.array([item[1] for item in candidates], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(names, p=probs))


def _build_segments(config: dict, n_steps: int, step_seconds: int, rng: np.random.Generator) -> list[Segment]:
    states_cfg = config["simulation"]["states"]
    segments: list[Segment] = []
    cursor = 0
    state = "idle"
    while cursor < n_steps:
        low, high = states_cfg[state]["duration_minutes"]
        minutes = float(rng.uniform(low, high))
        length = max(1, int(round(minutes * 60 / step_seconds)))
        stop = min(n_steps, cursor + length)
        segments.append(Segment(state=state, start=cursor, stop=stop))
        cursor = stop
        state = _choose_next_state(rng, state)
    return segments


def _triangle_wave(phase: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(2.0 * phase - 1.0)


def _segment_signals(
    state: str,
    n: int,
    step_seconds: int,
    rng: np.random.Generator,
    equipment: dict,
    depth_fraction: float,
    formation_hardness: float,
) -> dict[str, np.ndarray]:
    t = np.arange(n) * step_seconds
    progress = np.linspace(0.0, 1.0, n, endpoint=False)
    mud_rated = float(equipment["mud_pump_rated_kw"])
    top_rated = float(equipment["topdrive_rated_kw"])
    draw_continuous = float(equipment["drawworks_continuous_rated_kw"])
    draw_peak = float(equipment["drawworks_short_time_peak_kw"])
    depth_factor = 0.88 + 0.24 * float(np.clip(depth_fraction, 0.0, 1.0))
    formation = np.clip(
        formation_hardness + 0.035 * np.sin(2 * np.pi * t / 900.0), 0.70, 1.35
    )
    pump_spm = np.zeros(n)
    pressure_mpa = np.zeros(n)
    top_rpm = np.zeros(n)
    hookload_pct = np.full(n, 20.0 + 45.0 * depth_fraction)
    hook_speed = np.zeros(n)
    rop = np.zeros(n)

    if state == "idle":
        mud = mud_rated * 0.025 + 8 * np.sin(2 * np.pi * t / 180) + rng.normal(0, 3, n)
        top = top_rated * 0.014 + rng.normal(0, 2, n)
        draw = draw_continuous * 0.016 + rng.normal(0, 3, n)
        other = 95 + 12 * np.sin(2 * np.pi * t / 420) + rng.normal(0, 5, n)
        pump_spm = np.clip(4 + rng.normal(0, 1.2, n), 0, 10)
        pressure_mpa = np.clip(1.2 + rng.normal(0, 0.12, n), 0, None)
    elif state == "circulation":
        base = mud_rated * rng.uniform(0.46, 0.62) * depth_factor
        mud = base + 35 * np.sin(2 * np.pi * t / rng.uniform(100, 180)) + rng.normal(0, 9, n)
        top = top_rated * rng.uniform(0.08, 0.15) + 18 * np.sin(2 * np.pi * t / 75) + rng.normal(0, 5, n)
        draw = draw_continuous * 0.027 + rng.normal(0, 4, n)
        other = 135 + 18 * np.sin(2 * np.pi * t / 300) + rng.normal(0, 7, n)
        pump_spm = np.clip(72 + 42 * mud / mud_rated + rng.normal(0, 1.5, n), 65, 120)
        pressure_mpa = np.clip(4 + 29 * mud / mud_rated + 3.0 * depth_fraction, 5, 35)
        top_rpm = np.clip(35 + 18 * np.sin(2 * np.pi * t / 75) + rng.normal(0, 2, n), 5, 70)
    elif state == "drilling":
        mud = (
            mud_rated * rng.uniform(0.50, 0.66) * depth_factor * (0.93 + 0.08 * formation)
            + 35 * np.sin(2 * np.pi * t / 150)
            + rng.normal(0, 10, n)
        )
        top = (
            top_rated
            * float(np.clip(rng.normal(0.34, 0.025), 0.29, 0.40))
            * (0.65 + 0.50 * formation)
            + 55 * np.sin(2 * np.pi * t / 45)
            + rng.normal(0, 12, n)
        )
        draw = draw_continuous * 0.04 + 20 * np.sin(2 * np.pi * t / 95) + rng.normal(0, 6, n)
        other = 155 + 20 * np.sin(2 * np.pi * t / 260) + rng.normal(0, 8, n)
        pump_spm = np.clip(70 + 45 * mud / mud_rated + rng.normal(0, 1.8, n), 70, 120)
        pressure_mpa = np.clip(5 + 30 * mud / mud_rated + 4 * depth_fraction, 8, 38)
        top_rpm = np.clip(105 + 30 * np.sin(2 * np.pi * t / 45) + rng.normal(0, 4, n), 45, 180)
        hookload_pct = np.clip(48 + 38 * depth_fraction + rng.normal(0, 1.0, n), 35, 92)
        rop_base = rng.uniform(14.0, 28.0) * (1.0 - 0.30 * depth_fraction)
        rop = np.clip(rop_base / np.power(formation, 1.65) + rng.normal(0, 0.8, n), 3, 35)
    elif state == "connection":
        # 接立根阶段泥浆泵/旋转负荷下降，绞车出现数次短时动作。
        mud = mud_rated * (0.10 + 0.32 * (1.0 - np.sin(np.pi * progress))) + rng.normal(0, 9, n)
        top = top_rated * (0.05 + 0.05 * (1.0 - np.sin(np.pi * progress))) + rng.normal(0, 5, n)
        period = rng.uniform(80, 150)
        tri = _triangle_wave((t % period) / period)
        draw = draw_continuous * 0.08 + rng.uniform(0.36, 0.60) * draw_continuous * np.power(tri, 2.2) + rng.normal(0, 14, n)
        other = 145 + rng.normal(0, 8, n)
        pump_spm = np.clip(12 + 42 * mud / mud_rated, 8, 58)
        pressure_mpa = np.clip(2 + 25 * mud / mud_rated, 2, 16)
        top_rpm = np.clip(8 + 12 * (1.0 - np.sin(np.pi * progress)), 0, 25)
        hook_speed = 0.22 * np.sign(np.sin(2 * np.pi * t / period)) * np.power(tri, 1.5)
    elif state == "tripping":
        mud = mud_rated * 0.045 + rng.normal(0, 5, n)
        top = top_rated * 0.03 + rng.normal(0, 4, n)
        # 公开论文给出起下钻约5–7分钟变化，另一削峰论文给出2–4分钟脉冲周期。
        period = rng.uniform(300, 420)
        tri = _triangle_wave((t % period) / period)
        envelope = rng.uniform(0.80, 1.15) * (1.0 - 0.25 * progress)
        draw = draw_continuous * 0.06 + envelope * rng.uniform(0.62, 0.98) * draw_peak * np.power(tri, 2.8)
        draw += rng.normal(0, 18, n)
        other = 155 + 25 * np.sin(2 * np.pi * t / 210) + rng.normal(0, 8, n)
        hookload_pct = np.clip((35 + 55 * depth_fraction) * (1.0 - 0.28 * progress), 25, 95)
        direction = np.where((t % period) < period / 2.0, 1.0, -0.45)
        hook_speed = direction * rng.uniform(0.75, 1.15) * np.power(tri, 1.3)
        pressure_mpa = np.clip(1.5 + rng.normal(0, 0.15, n), 0, None)
    elif state == "maintenance":
        mud = mud_rated * 0.017 + rng.normal(0, 3, n)
        top = top_rated * 0.014 + rng.normal(0, 2, n)
        draw = draw_continuous * 0.013 + rng.normal(0, 3, n)
        other = 115 + 60 * (np.sin(2 * np.pi * t / 240) > 0.65) + rng.normal(0, 7, n)
        pressure_mpa = np.clip(0.8 + rng.normal(0, 0.1, n), 0, None)
    else:
        raise ValueError(f"未知仿真工况: {state}")

    return {
        "mud": np.maximum(mud, 0.0),
        "top": np.maximum(top, 0.0),
        "draw": np.clip(draw, 0.0, draw_peak),
        "other": np.maximum(other, 0.0),
        "pump_spm": np.maximum(pump_spm, 0.0),
        "pressure_mpa": np.maximum(pressure_mpa, 0.0),
        "top_rpm": np.maximum(top_rpm, 0.0),
        "hookload_pct": np.clip(hookload_pct, 0.0, 100.0),
        "hook_speed": hook_speed,
        "rop": np.maximum(rop, 0.0),
        "formation": formation,
    }


def _inject_missing_blocks(
    values: np.ndarray,
    days: int,
    step_seconds: int,
    probability_per_day: float,
    duration_seconds: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    observed = values.copy()
    missing = np.zeros(len(values), dtype=bool)
    steps_per_day = int(round(86400 / step_seconds))
    for day in range(days):
        if rng.random() > probability_per_day:
            continue
        day_start = day * steps_per_day
        day_stop = min(len(values), (day + 1) * steps_per_day)
        if day_stop - day_start < 10:
            continue
        start = int(rng.integers(day_start, day_stop - 1))
        seconds = int(rng.integers(duration_seconds[0], duration_seconds[1] + 1))
        length = max(1, int(round(seconds / step_seconds)))
        stop = min(day_stop, start + length)
        observed[start:stop] = np.nan
        missing[start:stop] = True
    return observed, missing


def generate_synthetic_rig_load(config: dict) -> pd.DataFrame:
    project = config["project"]
    sim = config["simulation"]
    seed = int(project["seed"])
    days = int(project["days"])
    step_seconds = int(project["frequency_seconds"])
    n_steps = int(days * 86400 / step_seconds)
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start=project["start_time"], periods=n_steps, freq=f"{step_seconds}s"
    )
    seconds = np.arange(n_steps) * step_seconds
    aux_cfg = sim["auxiliary"]
    daily = np.sin(2 * np.pi * seconds / 86400 - np.pi / 2)
    auxiliary = (
        float(aux_cfg["mean_kw"])
        + float(aux_cfg["daily_amplitude_kw"]) * daily
        + _colored_noise(
            rng,
            n_steps,
            float(aux_cfg["ar_noise_sigma_kw"]),
            float(aux_cfg["ar_phi"]),
        )
    )

    mud = np.zeros(n_steps)
    top = np.zeros(n_steps)
    draw = np.zeros(n_steps)
    other = np.zeros(n_steps)
    pump_spm = np.zeros(n_steps)
    pressure_mpa = np.zeros(n_steps)
    top_rpm = np.zeros(n_steps)
    hookload_pct = np.zeros(n_steps)
    hook_speed = np.zeros(n_steps)
    rop = np.zeros(n_steps)
    formation_index = np.zeros(n_steps)
    well_depth = np.zeros(n_steps)
    states = np.empty(n_steps, dtype=object)
    progress = np.zeros(n_steps)
    segments = _build_segments(config, n_steps, step_seconds, rng)
    equipment = sim["equipment"]
    well_cfg = sim["well"]
    current_depth = float(well_cfg["start_depth_m"])
    target_depth = float(well_cfg["target_depth_m"])
    current_formation = float(well_cfg["initial_formation_hardness_index"])
    for segment in segments:
        length = segment.stop - segment.start
        # Formation changes slowly by operational segment; depth evolves only while drilling.
        current_formation = float(np.clip(0.82 * current_formation + 0.18 * rng.normal(1.0, 0.18), 0.72, 1.35))
        depth_fraction = float(
            np.clip(
                (current_depth - float(well_cfg["start_depth_m"]))
                / max(target_depth - float(well_cfg["start_depth_m"]), 1.0),
                0.0,
                1.0,
            )
        )
        signals = _segment_signals(
            segment.state,
            length,
            step_seconds,
            rng,
            equipment,
            depth_fraction,
            current_formation,
        )
        selected = slice(segment.start, segment.stop)
        mud[selected] = signals["mud"]
        top[selected] = signals["top"]
        draw[selected] = signals["draw"]
        other[selected] = signals["other"]
        pump_spm[selected] = signals["pump_spm"]
        pressure_mpa[selected] = signals["pressure_mpa"]
        top_rpm[selected] = signals["top_rpm"]
        hookload_pct[selected] = signals["hookload_pct"]
        hook_speed[selected] = signals["hook_speed"]
        rop[selected] = signals["rop"]
        formation_index[selected] = signals["formation"]
        depth_increment = signals["rop"] * step_seconds / 3600.0
        well_depth[selected] = np.minimum(current_depth + np.cumsum(depth_increment), target_depth)
        current_depth = float(well_depth[segment.stop - 1])
        states[segment.start:segment.stop] = segment.state
        progress[segment.start:segment.stop] = np.linspace(0.0, 1.0, length, endpoint=False)

    physical_power = auxiliary + mud + top + draw + other
    # 传感器偏置缓慢漂移，叠加白噪声；保留 physical_power 供仿真质量核验。
    bias = _colored_noise(rng, n_steps, float(sim["sensor_bias_sigma_kw"]), 0.9995)
    measurement = physical_power + bias + rng.normal(
        0.0, float(sim["measurement_noise_sigma_kw"]), n_steps
    )
    quantization = float(sim.get("measurement_quantization_kw", 0.0))
    if quantization > 0.0:
        measurement = np.round(measurement / quantization) * quantization

    # 极少量设备切换尖峰/跌落，模拟接触器、变频器或突加/突卸负荷。
    transient = np.zeros(n_steps)
    event_count = max(8, days * int(sim.get("transient_events_per_day", 18)))
    for _ in range(event_count):
        center = int(rng.integers(2, n_steps - 3))
        width = int(rng.integers(1, max(2, int(30 / step_seconds) + 1)))
        amplitude = float(rng.uniform(-180, 300))
        idx = np.arange(max(0, center - width), min(n_steps, center + width + 1))
        transient[idx] += amplitude * np.exp(-0.5 * ((idx - center) / max(width / 2, 0.8)) ** 2)
    measurement += transient

    low, high = map(float, sim["total_power_clip_kw"])
    physical_power = np.clip(physical_power, low, high)
    measurement = np.clip(measurement, low, high)

    raw_measurement, missing = _inject_missing_blocks(
        measurement,
        days,
        step_seconds,
        float(sim["missing_block_probability_per_day"]),
        tuple(map(int, sim["missing_block_duration_seconds"])),
        rng,
    )
    imputed = pd.Series(raw_measurement).interpolate(limit_direction="both").to_numpy()
    state_changed = np.r_[True, states[1:] != states[:-1]]

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "total_active_power_kw": imputed.astype("float32"),
            "total_active_power_kw_raw": raw_measurement.astype("float32"),
            "physical_total_power_kw": physical_power.astype("float32"),
            "auxiliary_power_kw": auxiliary.astype("float32"),
            "mud_pump_power_kw": mud.astype("float32"),
            "topdrive_power_kw": top.astype("float32"),
            "drawworks_power_kw": draw.astype("float32"),
            "other_power_kw": other.astype("float32"),
            "well_depth_m": well_depth.astype("float32"),
            "formation_hardness_index": formation_index.astype("float32"),
            "pump_strokes_per_min": pump_spm.astype("float32"),
            "standpipe_pressure_mpa": pressure_mpa.astype("float32"),
            "topdrive_rpm": top_rpm.astype("float32"),
            "hookload_pct": hookload_pct.astype("float32"),
            "hook_speed_m_per_s": hook_speed.astype("float32"),
            "rate_of_penetration_m_per_h": rop.astype("float32"),
            "operation_state": states,
            "operation_state_code": np.vectorize(STATE_TO_CODE.get)(states).astype("int16"),
            "state_progress": progress.astype("float32"),
            "state_changed": state_changed,
            "transient_event_kw": transient.astype("float32"),
            "quality_flag": np.where(missing, "imputed", "observed"),
            "source_type": "synthetic",
        }
    )
    return data
