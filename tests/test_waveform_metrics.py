from __future__ import annotations

import json
import math

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity.waveform_metrics import (
    chirp_band_metrics,
    one_step_tracking_metrics,
    sine_response_metrics,
    state_state_metrics,
)


JOINTS = ("joint_a", "joint_b")
FRAMES = ("tip",)


def _run(
    *,
    backend: str,
    dt: float,
    duration: float,
    target,
    state,
    frame=None,
) -> AdapterRunResult:
    steps = round(duration / dt)
    samples: list[TraceSample] = []
    for step in range(steps + 1):
        time_s = step * dt
        joint_positions = {
            name: float(state(name, step, time_s)) for name in JOINTS
        }
        position_targets = {
            name: float(target(name, step, time_s)) for name in JOINTS
        }
        if frame is None:
            frame_poses = {"tip": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)}
        else:
            frame_poses = {"tip": tuple(float(value) for value in frame(step, time_s))}
        samples.append(
            TraceSample(
                step=step,
                time_s=time_s,
                qpos=tuple(joint_positions.values()),
                qvel=(0.0, 0.0),
                joint_positions=joint_positions,
                frame_poses=frame_poses,
                position_targets=position_targets,
                contact_count=0,
            )
        )
    return AdapterRunResult(
        backend=backend,
        scenario_id="synthetic",
        status="completed",
        message="synthetic",
        dt=dt,
        requested_steps=steps,
        completed_steps=steps,
        joint_names=JOINTS,
        frame_names=FRAMES,
        samples=tuple(samples),
    )


def _assert_strict_json(value: object) -> None:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == value


def test_state_state_metrics_report_rmse_p95_and_first_peak() -> None:
    reference = _run(
        backend="mujoco",
        dt=1.0,
        duration=2.0,
        target=lambda *_: 0.0,
        state=lambda *_: 0.0,
    )
    joint_errors = {
        "joint_a": (0.0, 1.0, 2.0),
        "joint_b": (0.0, -2.0, 0.0),
    }
    candidate = _run(
        backend="ovphysx",
        dt=1.0,
        duration=2.0,
        target=lambda *_: 0.0,
        state=lambda name, step, _time: joint_errors[name][step],
        frame=lambda step, _time: (
            float(step),
            0.0,
            0.0,
            math.cos(step * math.pi / 8.0),
            0.0,
            0.0,
            math.sin(step * math.pi / 8.0),
        ),
    )

    metrics = state_state_metrics(
        reference, candidate, joint_names=JOINTS, frame_names=FRAMES
    )

    assert metrics["joint"]["rmse_rad"] == pytest.approx(math.sqrt(1.5))
    assert metrics["joint"]["max_abs_rad"] == 2.0
    assert metrics["joint"]["per_entity"]["joint_a"] == {
        "max_abs_rad": 2.0,
        "peak_step": 2,
        "peak_time_s": 2.0,
    }
    # Tied peaks resolve to the first trace sample.
    assert metrics["joint"]["per_entity"]["joint_b"]["peak_step"] == 1
    assert metrics["frame_position"]["max_distance_m"] == 2.0
    assert metrics["frame_orientation"]["max_angle_rad"] == pytest.approx(math.pi / 2.0)
    _assert_strict_json(metrics)


def test_one_step_tracking_uses_target_k_and_subsequent_state() -> None:
    targets = (0.0, 1.0, 2.0)
    states = (10.0, 0.5, 1.5)
    run = _run(
        backend="mujoco",
        dt=1.0,
        duration=2.0,
        target=lambda _name, step, _time: targets[step],
        state=lambda _name, step, _time: states[step],
    )

    metrics = one_step_tracking_metrics(
        run, joint_names=JOINTS, frame_names=FRAMES
    )

    assert metrics["pairing"] == "target[k]_to_state[k+1]"
    assert metrics["command_interval_count"] == 2
    assert metrics["rmse_rad"] == 0.5
    assert metrics["max_abs_rad"] == 0.5
    peak = metrics["per_entity"]["joint_a"]
    assert peak["target_step"] == 0
    assert peak["response_step"] == 1
    _assert_strict_json(metrics)

    # Frame data can remain present when a joint-only metric does not request it.
    joint_only = one_step_tracking_metrics(run, joint_names=JOINTS)
    assert joint_only == metrics


def test_sine_response_fits_true_input_and_response_times() -> None:
    frequency = 1.0
    input_amplitude = 0.025
    input_phase = 0.7
    output_amplitude = 0.0125
    lag = 0.2
    run = _run(
        backend="mujoco",
        dt=0.01,
        duration=4.0,
        target=lambda _name, _step, time_s: (
            0.025 + input_amplitude * math.cos(math.tau * frequency * time_s + input_phase)
        ),
        state=lambda _name, _step, time_s: (
            0.02
            + output_amplitude
            * math.cos(math.tau * frequency * time_s + input_phase - lag)
        ),
    )

    metrics = sine_response_metrics(
        run,
        joint_names=JOINTS,
        frame_names=FRAMES,
        frequency_hz=frequency,
        command_amplitude_rad=input_amplitude,
    )

    assert metrics["command_interval_count"] == 300
    joint = metrics["per_joint"]["joint_a"]
    assert joint["input"]["amplitude_rad"] == pytest.approx(input_amplitude, abs=1e-12)
    assert joint["output"]["amplitude_rad"] == pytest.approx(output_amplitude, abs=1e-12)
    assert joint["gain_vs_registered_command_amplitude"] == pytest.approx(0.5)
    assert joint["phase_available"] is True
    assert joint["phase_lag_rad"] == pytest.approx(lag, abs=1e-12)
    assert joint["input"]["fit_residual_rmse_rad"] < 1e-12
    assert joint["output"]["fit_residual_rmse_rad"] < 1e-12
    _assert_strict_json(metrics)


def test_sine_low_output_amplitude_uses_none_not_nan_for_phase() -> None:
    run = _run(
        backend="mujoco",
        dt=0.01,
        duration=4.0,
        target=lambda _name, _step, time_s: 0.025
        - 0.025 * math.cos(math.tau * time_s),
        state=lambda *_: 0.0,
    )

    metrics = sine_response_metrics(
        run, joint_names=JOINTS, frame_names=FRAMES
    )
    joint = metrics["per_joint"]["joint_a"]

    assert joint["output"]["amplitude_rad"] < 1e-12
    assert joint["phase_available"] is False
    assert joint["phase_lag_rad"] is None
    _assert_strict_json(metrics)


def test_chirp_bands_use_interval_midpoint_and_report_tracking_and_crosssim() -> None:
    dt = 0.1
    duration = 4.0

    def band_value(step: int) -> float:
        if step == 0:
            return 0.0
        midpoint = (step - 0.5) * dt
        frequency = 0.5 + 0.875 * midpoint
        if frequency < 1.0:
            return 1.0
        if frequency < 2.0:
            return 2.0
        if frequency < 3.0:
            return 3.0
        return 4.0

    reference = _run(
        backend="mujoco",
        dt=dt,
        duration=duration,
        target=lambda *_: 0.0,
        state=lambda _name, step, _time: band_value(step),
        frame=lambda step, _time: (band_value(step), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    candidate = _run(
        backend="ovphysx",
        dt=dt,
        duration=duration,
        target=lambda *_: 0.0,
        state=lambda _name, step, _time: 1.1 * band_value(step),
        frame=lambda step, _time: (
            1.1 * band_value(step),
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    metrics = chirp_band_metrics(
        reference,
        candidate,
        joint_names=JOINTS,
        frame_names=FRAMES,
    )

    assert len(metrics["bands"]) == 4
    for index, record in enumerate(metrics["bands"], start=1):
        assert record["command_interval_count"] > 0
        assert record["tracking"]["reference"]["metrics"]["rmse_rad"] == pytest.approx(
            float(index)
        )
        assert record["tracking"]["candidate"]["metrics"]["max_abs_rad"] == pytest.approx(
            1.1 * index
        )
        assert record["crosssim"]["joint"]["max_abs_rad"] == pytest.approx(
            0.1 * index
        )
        assert record["crosssim"]["frame_position"]["max_distance_m"] == pytest.approx(
            0.1 * index
        )
    assert metrics["bands"][-1]["upper_hz_inclusive"] is True
    _assert_strict_json(metrics)


def test_metrics_fail_closed_on_misalignment_nonfinite_and_empty_band() -> None:
    reference = _run(
        backend="mujoco",
        dt=1.0,
        duration=2.0,
        target=lambda *_: 0.0,
        state=lambda *_: 0.0,
    )
    candidate = _run(
        backend="ovphysx",
        dt=0.5,
        duration=2.0,
        target=lambda *_: 0.0,
        state=lambda *_: 0.0,
    )
    with pytest.raises(ValueError, match="same sample count"):
        state_state_metrics(
            reference, candidate, joint_names=JOINTS, frame_names=FRAMES
        )

    bad_samples = list(reference.samples)
    bad = bad_samples[1]
    bad_samples[1] = TraceSample(
        step=bad.step,
        time_s=bad.time_s,
        qpos=bad.qpos,
        qvel=bad.qvel,
        joint_positions={"joint_a": math.nan, "joint_b": 0.0},
        frame_poses=bad.frame_poses,
        position_targets=bad.position_targets,
        contact_count=bad.contact_count,
    )
    nonfinite = AdapterRunResult(
        backend=reference.backend,
        scenario_id=reference.scenario_id,
        status=reference.status,
        message=reference.message,
        dt=reference.dt,
        requested_steps=reference.requested_steps,
        completed_steps=reference.completed_steps,
        joint_names=reference.joint_names,
        frame_names=reference.frame_names,
        samples=tuple(bad_samples),
    )
    with pytest.raises(ValueError, match="finite number"):
        one_step_tracking_metrics(
            nonfinite, joint_names=JOINTS, frame_names=FRAMES
        )

    with pytest.raises(ValueError, match="contains no command intervals"):
        chirp_band_metrics(
            reference,
            reference,
            joint_names=JOINTS,
            frame_names=FRAMES,
            bands_hz=((10.0, 11.0),),
        )

    too_long = _run(
        backend="mujoco",
        dt=0.1,
        duration=5.0,
        target=lambda *_: 0.0,
        state=lambda *_: 0.0,
    )
    with pytest.raises(ValueError, match="cover every applied command interval"):
        chirp_band_metrics(
            too_long,
            too_long,
            joint_names=JOINTS,
            frame_names=FRAMES,
        )
