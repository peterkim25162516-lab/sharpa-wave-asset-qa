"""Descriptive metrics for preregistered no-contact waveform trajectories.

This module deliberately contains no acceptance thresholds or scientific label
logic.  Its inputs are expected to have passed the structural evidence
validator already; the checks below only make the numerical calculations fail
closed instead of producing partial or non-finite JSON.

The recorded command/state timing contract is important: ``target[k]`` is the
command applied over ``[t_k, t_{k+1})``, while ``state[k]`` is the observation
at ``t_k`` before that command has advanced the simulator.  Tracking and
response calculations therefore pair ``target[k]`` with ``state[k + 1]``.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample


SINE_PHASE_AMPLITUDE_FLOOR_RAD = 1e-6
DEFAULT_CHIRP_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
)
_TIMESTAMP_SCALE = 1_000_000_000_000


def _names(values: Sequence[str], context: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    names = tuple(values)
    if not allow_empty and not names:
        raise ValueError(f"{context} must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{context} must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError(f"{context} must not contain duplicates")
    return names


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _validate_run(
    run: AdapterRunResult,
    joint_names: tuple[str, ...],
    frame_names: tuple[str, ...],
) -> None:
    if not isinstance(run, AdapterRunResult):
        raise TypeError("run must be an AdapterRunResult")
    if not run.completed:
        raise ValueError("waveform metrics require a completed run")
    dt_s = _finite(run.dt, "run.dt")
    if dt_s <= 0.0:
        raise ValueError("run.dt must be positive")
    if tuple(run.joint_names) != joint_names:
        raise ValueError("run joint_names do not match the requested canonical order")
    if frame_names and tuple(run.frame_names) != frame_names:
        raise ValueError("run frame_names do not match the requested canonical order")
    if len(run.samples) != run.requested_steps + 1:
        raise ValueError("run must contain one initial sample plus every requested step")
    if not run.samples:
        raise ValueError("run samples must not be empty")
    prior_time_key: int | None = None
    for index, sample in enumerate(run.samples):
        if not isinstance(sample, TraceSample):
            raise ValueError("run samples must contain TraceSample values")
        if sample.step != index:
            raise ValueError("run sample steps must be contiguous from zero")
        time_s = _finite(sample.time_s, f"sample[{index}].time_s")
        time_key = round(time_s * _TIMESTAMP_SCALE)
        if time_key != round(index * dt_s * _TIMESTAMP_SCALE):
            raise ValueError("run sample timestamps do not match step * dt")
        if prior_time_key is not None and time_key <= prior_time_key:
            raise ValueError("run sample timestamps must be strictly increasing")
        prior_time_key = time_key
        if set(sample.joint_positions) != set(joint_names):
            raise ValueError("sample joint positions do not exactly cover requested joints")
        if set(sample.position_targets) != set(joint_names):
            raise ValueError("sample position targets do not exactly cover requested joints")
        if frame_names and set(sample.frame_poses) != set(frame_names):
            raise ValueError("sample frame poses do not exactly cover requested frames")
        for name in joint_names:
            _finite(sample.joint_positions[name], f"sample[{index}].joint_positions[{name!r}]")
            _finite(sample.position_targets[name], f"sample[{index}].position_targets[{name!r}]")
        for name in frame_names:
            pose = sample.frame_poses[name]
            if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)) or len(pose) != 7:
                raise ValueError(f"sample[{index}].frame_poses[{name!r}] must have seven values")
            for component_index, value in enumerate(pose):
                _finite(value, f"sample[{index}].frame_poses[{name!r}][{component_index}]")


def _aligned_samples(
    reference: AdapterRunResult,
    candidate: AdapterRunResult,
    joint_names: tuple[str, ...],
    frame_names: tuple[str, ...],
) -> tuple[tuple[TraceSample, TraceSample], ...]:
    _validate_run(reference, joint_names, frame_names)
    _validate_run(candidate, joint_names, frame_names)
    if reference.scenario_id != candidate.scenario_id:
        raise ValueError("state-state traces must describe the same scenario")
    if len(reference.samples) != len(candidate.samples):
        raise ValueError("state-state traces must have the same sample count")
    aligned: list[tuple[TraceSample, TraceSample]] = []
    for reference_sample, candidate_sample in zip(reference.samples, candidate.samples):
        if reference_sample.step != candidate_sample.step:
            raise ValueError("state-state traces must have identical sample steps")
        if round(reference_sample.time_s * _TIMESTAMP_SCALE) != round(
            candidate_sample.time_s * _TIMESTAMP_SCALE
        ):
            raise ValueError("state-state traces must align at picosecond-rounded timestamps")
        aligned.append((reference_sample, candidate_sample))
    return tuple(aligned)


def _quaternion_angle(pose_a: Sequence[float], pose_b: Sequence[float]) -> float:
    qa = np.asarray(tuple(float(value) for value in pose_a[3:7]), dtype=float)
    qb = np.asarray(tuple(float(value) for value in pose_b[3:7]), dtype=float)
    norm_a = float(np.linalg.norm(qa))
    norm_b = float(np.linalg.norm(qb))
    if not math.isfinite(norm_a) or not math.isfinite(norm_b) or norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("frame pose contains a zero or non-finite quaternion")
    qa /= norm_a
    qb /= norm_b
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    difference_norm = float(np.linalg.norm(qa - qb))
    sum_norm = float(np.linalg.norm(qa + qb))
    angle = 4.0 * math.atan2(difference_norm, sum_norm)
    if not math.isfinite(angle):
        raise ValueError("frame orientation delta is non-finite")
    return angle


def _metric_summary(
    values: np.ndarray,
    *,
    entity_names: tuple[str, ...],
    steps: Sequence[int],
    times_s: Sequence[float],
    rmse_key: str,
    p95_key: str,
    max_key: str,
) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape != (len(steps), len(entity_names)):
        raise ValueError("metric array shape does not match samples and entities")
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError("metric values must be non-empty and finite")
    absolute = np.abs(array)
    flat = absolute.reshape(-1)
    per_entity: dict[str, object] = {}
    for entity_index, name in enumerate(entity_names):
        column = absolute[:, entity_index]
        peak_index = int(np.argmax(column))
        per_entity[name] = {
            max_key: float(column[peak_index]),
            "peak_step": int(steps[peak_index]),
            "peak_time_s": float(times_s[peak_index]),
        }
    return {
        "sample_count": int(array.shape[0]),
        "entity_count": int(array.shape[1]),
        "element_count": int(array.size),
        rmse_key: float(math.sqrt(float(np.mean(np.square(array))))),
        p95_key: float(np.percentile(flat, 95.0, method="linear")),
        max_key: float(np.max(flat)),
        "per_entity": per_entity,
    }


def _state_state_metrics_from_pairs(
    pairs: Sequence[tuple[TraceSample, TraceSample]],
    joint_names: tuple[str, ...],
    frame_names: tuple[str, ...],
) -> dict[str, object]:
    if not pairs:
        raise ValueError("state-state metrics require at least one aligned sample")
    steps = [reference.step for reference, _ in pairs]
    times = [float(reference.time_s) for reference, _ in pairs]
    joint_errors = np.asarray(
        [
            [
                float(candidate.joint_positions[name])
                - float(reference.joint_positions[name])
                for name in joint_names
            ]
            for reference, candidate in pairs
        ],
        dtype=float,
    )
    result: dict[str, object] = {
        "sample_count": len(pairs),
        "joint": _metric_summary(
            joint_errors,
            entity_names=joint_names,
            steps=steps,
            times_s=times,
            rmse_key="rmse_rad",
            p95_key="p95_abs_rad",
            max_key="max_abs_rad",
        ),
    }
    if frame_names:
        position_errors = np.asarray(
            [
                [
                    math.sqrt(
                        sum(
                            (
                                float(candidate.frame_poses[name][component])
                                - float(reference.frame_poses[name][component])
                            )
                            ** 2
                            for component in range(3)
                        )
                    )
                    for name in frame_names
                ]
                for reference, candidate in pairs
            ],
            dtype=float,
        )
        orientation_errors = np.asarray(
            [
                [
                    _quaternion_angle(
                        reference.frame_poses[name], candidate.frame_poses[name]
                    )
                    for name in frame_names
                ]
                for reference, candidate in pairs
            ],
            dtype=float,
        )
        result["frame_position"] = _metric_summary(
            position_errors,
            entity_names=frame_names,
            steps=steps,
            times_s=times,
            rmse_key="rmse_m",
            p95_key="p95_distance_m",
            max_key="max_distance_m",
        )
        result["frame_orientation"] = _metric_summary(
            orientation_errors,
            entity_names=frame_names,
            steps=steps,
            times_s=times,
            rmse_key="rmse_rad",
            p95_key="p95_angle_rad",
            max_key="max_angle_rad",
        )
    else:
        result["frame_position"] = None
        result["frame_orientation"] = None
    return result


def state_state_metrics(
    reference: AdapterRunResult,
    candidate: AdapterRunResult,
    *,
    joint_names: Sequence[str],
    frame_names: Sequence[str],
) -> dict[str, object]:
    """Return descriptive state deltas at aligned physical timestamps."""

    joints = _names(joint_names, "joint_names")
    frames = _names(frame_names, "frame_names", allow_empty=True)
    pairs = _aligned_samples(reference, candidate, joints, frames)
    return _state_state_metrics_from_pairs(pairs, joints, frames)


def _tracking_metrics_from_indices(
    run: AdapterRunResult,
    joint_names: tuple[str, ...],
    interval_indices: Sequence[int],
) -> dict[str, object]:
    if not interval_indices:
        raise ValueError("tracking metrics require at least one command interval")
    errors: list[list[float]] = []
    response_steps: list[int] = []
    response_times: list[float] = []
    for interval_index in interval_indices:
        if interval_index < 0 or interval_index + 1 >= len(run.samples):
            raise ValueError("tracking interval index is outside the run")
        target_sample = run.samples[interval_index]
        response_sample = run.samples[interval_index + 1]
        errors.append(
            [
                float(response_sample.joint_positions[name])
                - float(target_sample.position_targets[name])
                for name in joint_names
            ]
        )
        response_steps.append(response_sample.step)
        response_times.append(float(response_sample.time_s))
    summary = _metric_summary(
        np.asarray(errors, dtype=float),
        entity_names=joint_names,
        steps=response_steps,
        times_s=response_times,
        rmse_key="rmse_rad",
        p95_key="p95_abs_rad",
        max_key="max_abs_rad",
    )
    summary["command_interval_count"] = len(interval_indices)
    summary["pairing"] = "target[k]_to_state[k+1]"
    per_entity = summary["per_entity"]
    assert isinstance(per_entity, dict)
    for name, record in per_entity.items():
        assert isinstance(name, str) and isinstance(record, dict)
        response_step = int(record["peak_step"])
        record["target_step"] = response_step - 1
        record["response_step"] = response_step
    return summary


def one_step_tracking_metrics(
    run: AdapterRunResult,
    *,
    joint_names: Sequence[str],
    frame_names: Sequence[str] = (),
) -> dict[str, object]:
    """Measure ``state[k+1] - target[k]`` over every applied interval."""

    joints = _names(joint_names, "joint_names")
    frames = _names(frame_names, "frame_names", allow_empty=True)
    _validate_run(run, joints, frames)
    return _tracking_metrics_from_indices(run, joints, range(len(run.samples) - 1))


def _fit_sinusoid(
    times_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
) -> dict[str, float]:
    if times_s.ndim != 1 or values.ndim != 1 or times_s.shape != values.shape:
        raise ValueError("sinusoid fit times and values must be matching vectors")
    if times_s.size < 3 or not bool(np.isfinite(times_s).all()) or not bool(np.isfinite(values).all()):
        raise ValueError("sinusoid fit requires at least three finite observations")
    omega = math.tau * frequency_hz
    design = np.column_stack(
        (np.ones(times_s.size), np.cos(omega * times_s), np.sin(omega * times_s))
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != 3 or not bool(np.isfinite(coefficients).all()):
        raise ValueError("sinusoid fit design is rank deficient")
    predicted = design @ coefficients
    residual_rmse = math.sqrt(float(np.mean(np.square(values - predicted))))
    offset, cosine, sine = (float(value) for value in coefficients)
    amplitude = math.hypot(cosine, sine)
    # Re(C * exp(i*w*t)) == cosine*cos(w*t) + sine*sin(w*t).
    complex_real = cosine
    complex_imag = -sine
    phase = math.atan2(complex_imag, complex_real)
    return {
        "offset_rad": offset,
        "cosine_coefficient_rad": cosine,
        "sine_coefficient_rad": sine,
        "complex_real_rad": complex_real,
        "complex_imag_rad": complex_imag,
        "amplitude_rad": amplitude,
        "phase_rad": phase,
        "fit_residual_rmse_rad": residual_rmse,
    }


def _wrap_angle(value: float) -> float:
    wrapped = (value + math.pi) % math.tau - math.pi
    # Use +pi rather than -pi for the one ambiguous endpoint.
    return math.pi if wrapped == -math.pi and value > 0.0 else wrapped


def sine_response_metrics(
    run: AdapterRunResult,
    *,
    joint_names: Sequence[str],
    frame_names: Sequence[str] = (),
    frequency_hz: float = 1.0,
    command_amplitude_rad: float = 0.025,
    window_start_s: float = 1.0,
    window_end_s: float = 4.0,
    phase_amplitude_floor_rad: float = SINE_PHASE_AMPLITUDE_FLOOR_RAD,
) -> dict[str, object]:
    """Fit per-joint fundamental response over applied command intervals.

    Input samples use ``(t_k, target[k])`` and output samples use
    ``(t_{k+1}, state[k+1])``.  Phase lag is positive when the fitted output
    phase trails the fitted input phase.  Low-output phase is represented as
    ``None``; no NaN or Infinity is emitted.
    """

    joints = _names(joint_names, "joint_names")
    frames = _names(frame_names, "frame_names", allow_empty=True)
    _validate_run(run, joints, frames)
    frequency = _finite(frequency_hz, "frequency_hz")
    amplitude = _finite(command_amplitude_rad, "command_amplitude_rad")
    start = _finite(window_start_s, "window_start_s")
    end = _finite(window_end_s, "window_end_s")
    floor = _finite(phase_amplitude_floor_rad, "phase_amplitude_floor_rad")
    if frequency <= 0.0 or amplitude <= 0.0 or floor < 0.0 or not start < end:
        raise ValueError("sine fit frequency/amplitude/window/floor is invalid")
    interval_indices = [
        index
        for index, sample in enumerate(run.samples[:-1])
        if start <= float(sample.time_s) < end
    ]
    if len(interval_indices) < 3:
        raise ValueError("sine fit window contains fewer than three command intervals")
    input_times = np.asarray(
        [run.samples[index].time_s for index in interval_indices], dtype=float
    )
    output_times = np.asarray(
        [run.samples[index + 1].time_s for index in interval_indices], dtype=float
    )
    per_joint: dict[str, object] = {}
    for name in joints:
        input_values = np.asarray(
            [run.samples[index].position_targets[name] for index in interval_indices],
            dtype=float,
        )
        output_values = np.asarray(
            [run.samples[index + 1].joint_positions[name] for index in interval_indices],
            dtype=float,
        )
        input_fit = _fit_sinusoid(input_times, input_values, frequency)
        output_fit = _fit_sinusoid(output_times, output_values, frequency)
        input_phase = input_fit["phase_rad"]
        output_phase = output_fit["phase_rad"]
        phase_available = (
            input_fit["amplitude_rad"] >= floor
            and output_fit["amplitude_rad"] >= floor
        )
        per_joint[name] = {
            "input": input_fit,
            "output": output_fit,
            "gain_vs_registered_command_amplitude": (
                output_fit["amplitude_rad"] / amplitude
            ),
            "complex_coefficient_delta_rad": {
                "real": output_fit["complex_real_rad"] - input_fit["complex_real_rad"],
                "imag": output_fit["complex_imag_rad"] - input_fit["complex_imag_rad"],
            },
            "phase_available": phase_available,
            "phase_lag_rad": (
                _wrap_angle(input_phase - output_phase) if phase_available else None
            ),
        }
    return {
        "frequency_hz": frequency,
        "registered_command_amplitude_rad": amplitude,
        "phase_amplitude_floor_rad": floor,
        "window": {"start_s_inclusive": start, "end_s_exclusive": end},
        "command_interval_count": len(interval_indices),
        "timing": {
            "input": "(t_k,target[k])",
            "output": "(t_{k+1},state[k+1])",
        },
        "per_joint": per_joint,
    }


def _validated_chirp_bands(
    bands_hz: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    bands: list[tuple[float, float]] = []
    for index, raw in enumerate(bands_hz):
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise ValueError(f"chirp band {index} must contain lower and upper bounds")
        lower = _finite(raw[0], f"chirp band {index} lower")
        upper = _finite(raw[1], f"chirp band {index} upper")
        if lower < 0.0 or not lower < upper:
            raise ValueError(f"chirp band {index} bounds are invalid")
        if bands and not math.isclose(lower, bands[-1][1], rel_tol=0.0, abs_tol=0.0):
            raise ValueError("chirp bands must be ordered and exactly contiguous")
        bands.append((lower, upper))
    if not bands:
        raise ValueError("chirp bands must not be empty")
    return tuple(bands)


def chirp_band_metrics(
    reference: AdapterRunResult,
    candidate: AdapterRunResult,
    *,
    joint_names: Sequence[str],
    frame_names: Sequence[str],
    start_frequency_hz: float = 0.5,
    chirp_rate_hz_per_s: float = 0.875,
    bands_hz: Iterable[tuple[float, float]] = DEFAULT_CHIRP_BANDS_HZ,
) -> dict[str, object]:
    """Return fixed instantaneous-frequency-band tracking and state deltas.

    An applied interval is assigned by its midpoint frequency
    ``f(t_mid) = f0 + rate*t_mid``.  The last band includes its upper endpoint;
    all earlier bands are half-open.  Empty registered bands fail closed.
    """

    joints = _names(joint_names, "joint_names")
    frames = _names(frame_names, "frame_names", allow_empty=True)
    pairs = _aligned_samples(reference, candidate, joints, frames)
    f0 = _finite(start_frequency_hz, "start_frequency_hz")
    rate = _finite(chirp_rate_hz_per_s, "chirp_rate_hz_per_s")
    if f0 < 0.0 or rate <= 0.0:
        raise ValueError("chirp start frequency and rate must be non-negative/positive")
    bands = _validated_chirp_bands(bands_hz)
    band_records: list[dict[str, object]] = []
    assigned_indices: set[int] = set()
    for band_index, (lower, upper) in enumerate(bands):
        is_last = band_index == len(bands) - 1
        indices: list[int] = []
        for interval_index in range(len(reference.samples) - 1):
            start_time = float(reference.samples[interval_index].time_s)
            end_time = float(reference.samples[interval_index + 1].time_s)
            midpoint_frequency = f0 + rate * (start_time + end_time) / 2.0
            in_band = lower <= midpoint_frequency < upper
            if is_last and math.isclose(midpoint_frequency, upper, rel_tol=0.0, abs_tol=1e-12):
                in_band = True
            if in_band:
                if interval_index in assigned_indices:
                    raise ValueError("chirp interval was assigned to more than one band")
                indices.append(interval_index)
                assigned_indices.add(interval_index)
        if not indices:
            raise ValueError(f"chirp band [{lower}, {upper}] contains no command intervals")
        response_pairs = [pairs[index + 1] for index in indices]
        band_records.append(
            {
                "lower_hz_inclusive": lower,
                "upper_hz": upper,
                "upper_hz_inclusive": is_last,
                "command_interval_count": len(indices),
                "tracking": {
                    "reference": {
                        "backend": reference.backend,
                        "metrics": _tracking_metrics_from_indices(reference, joints, indices),
                    },
                    "candidate": {
                        "backend": candidate.backend,
                        "metrics": _tracking_metrics_from_indices(candidate, joints, indices),
                    },
                },
                "crosssim": _state_state_metrics_from_pairs(
                    response_pairs, joints, frames
                ),
            }
        )
    expected_interval_count = len(reference.samples) - 1
    if len(assigned_indices) != expected_interval_count:
        raise ValueError(
            "chirp bands do not cover every applied command interval"
        )
    return {
        "start_frequency_hz": f0,
        "chirp_rate_hz_per_s": rate,
        "assignment": "interval_midpoint_instantaneous_frequency",
        "timing": "target[k]_to_state[k+1]",
        "bands": band_records,
    }


__all__ = [
    "DEFAULT_CHIRP_BANDS_HZ",
    "SINE_PHASE_AMPLITUDE_FLOOR_RAD",
    "chirp_band_metrics",
    "one_step_tracking_metrics",
    "sine_response_metrics",
    "state_state_metrics",
]
