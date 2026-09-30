from __future__ import annotations

from copy import copy
from dataclasses import replace
import math
from pathlib import Path

import pytest

from wave_asset_qa.contact.metrics import (
    _quaternion_angle,
    compare_condition_effects,
    compare_trace_records,
    debounce_pair_activity,
    extract_run_metrics,
)
from wave_asset_qa.contact.records import (
    ContactCaseRecord,
    ContactExecutionRecord,
    ContactRun,
    ContactTraceSample,
)
from wave_asset_qa.contact.scenarios import expand_contact_cases, load_contact_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "contact_c0.json"


def _sample(step: int, dt_s: float, *, active: bool, gap_m: float) -> ContactTraceSample:
    sample = object.__new__(ContactTraceSample)
    joints = {f"left_joint_{index:02d}": 0.0 for index in range(22)}
    frames = {
        f"left_frame_{index}": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        for index in range(5)
    }
    for name, value in (
        ("step", step),
        ("time_s", step * dt_s),
        ("joint_positions", joints),
        ("joint_velocities", joints),
        ("frame_poses", frames),
        ("position_targets", joints),
        ("probe_center_world_m", (0.0, 0.0, 0.135 + gap_m)),
        ("signed_gap_m", gap_m),
        ("pair_active", active),
        ("raw_contact_count", None),
        ("native_contact_observation", None),
    ):
        object.__setattr__(sample, name, value)
    return sample


def _run(condition: str, dt_s: float = 0.002) -> ContactRun:
    manifest = load_contact_manifest(MANIFEST_PATH)
    variant = "base" if dt_s == 0.002 else "halved"
    case = next(
        case
        for case in expand_contact_cases(manifest)
        if case.simulator.value == "mujoco"
        and case.hand.value == "left"
        and case.condition.value == condition
        and case.timestep_variant.value == variant
        and case.repeat_index == 1
    )
    count = round(7.0 / dt_s)
    samples = []
    for step in range(count + 1):
        time_s = step * dt_s
        active = condition == "contact" and step > 0 and 1.0 <= time_s < 5.0
        if time_s < 0.5 or time_s >= 6.5:
            gap = 0.060
        elif 3.25 <= time_s < 3.75:
            gap = 0.0 if condition == "contact" else -0.010
        else:
            gap = 0.0
        samples.append(_sample(step, dt_s, active=active, gap_m=gap))
    run = object.__new__(ContactRun)
    record = ContactCaseRecord.from_dict(case.to_dict())
    execution = ContactExecutionRecord("completed", None, count, count)
    for name, value in (
        ("schema_version", 1),
        ("manifest_id", manifest.manifest_id),
        ("manifest_sha256", "0" * 64),
        ("case", record),
        ("execution", execution),
        ("mapping", {}),
        ("fixture_readback", {}),
        ("contact_observation", {}),
        ("samples", tuple(samples)),
        ("provenance", {}),
    ):
        object.__setattr__(run, name, value)
    return run


def _reference_qwxyz_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _axis_angle_qwxyz(
    axis: tuple[float, float, float], angle_rad: float
) -> tuple[float, float, float, float]:
    half = angle_rad / 2.0
    scale = math.sin(half)
    return (
        math.cos(half),
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
    )


def _with_frame_orientation(
    run: ContactRun, quaternion_qwxyz: tuple[float, float, float, float]
) -> ContactRun:
    cloned = copy(run)
    samples = tuple(
        replace(
            sample,
            frame_poses={
                name: (0.0, 0.0, 0.0, *quaternion_qwxyz)
                for name in sample.frame_poses
            },
        )
        for sample in run.samples
    )
    object.__setattr__(cloned, "samples", samples)
    return cloned


def test_debounce_uses_consecutive_intervals_and_backdates_transition() -> None:
    bits = [False, True, False, True, True, True, False, True, False, False]
    samples = tuple(
        _sample(index, 0.002, active=bit, gap_m=0.0)
        for index, bit in enumerate(bits)
    )

    activity = debounce_pair_activity(samples, dt_s=0.002)

    assert activity.required_consecutive_intervals == 2
    assert activity.interval_bits == (
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    )
    assert activity.onset_interval is not None
    assert activity.onset_interval.first_sample_index == 3
    assert activity.onset_interval.start_s == pytest.approx(0.004)
    assert activity.release_interval is not None
    assert activity.release_interval.first_sample_index == 8


def test_contact_and_sham_metrics_use_frozen_hold_and_recovery_windows() -> None:
    contact = extract_run_metrics(_run("contact"))
    sham = extract_run_metrics(_run("sham"))

    assert contact.onset_interval is not None
    assert contact.onset_interval.start_s == pytest.approx(0.998)
    assert contact.onset_interval.end_s == pytest.approx(1.0)
    assert contact.release_interval is not None
    assert contact.release_interval.start_s == pytest.approx(4.998)
    assert contact.hold_duty == 1.0
    assert contact.recovery_duty == 0.0
    assert contact.initial_gap_m == pytest.approx(0.060)
    assert contact.recovery_min_gap_m == pytest.approx(0.060)
    assert sham.onset_interval is None
    assert sham.hold_duty == 0.0
    assert sham.steady_hold_mean_gap_m == pytest.approx(-0.010)


def test_window_membership_uses_canonical_integer_sample_indices() -> None:
    run = _run("contact")
    boundary_step = round(3.25 / run.case.dt_s)
    samples = list(run.samples)
    # A valid serialized timestamp may differ from step*dt by sub-nanosecond
    # roundoff.  That integrity tolerance must not change which samples enter
    # the frozen [3.25, 3.75) steady-hold window.
    samples[boundary_step] = replace(
        samples[boundary_step],
        time_s=3.25 - 5.0e-10,
        signed_gap_m=0.25,
    )
    object.__setattr__(run, "samples", tuple(samples))

    metrics = extract_run_metrics(run)

    expected_sample_count = round((3.75 - 3.25) / run.case.dt_s)
    assert metrics.steady_hold_mean_gap_m == pytest.approx(
        0.25 / expected_sample_count
    )


def test_repeat_dt_and_contact_minus_sham_deltas_are_portable() -> None:
    base_contact = _run("contact", 0.002)
    base_sham = _run("sham", 0.002)
    half_contact = _run("contact", 0.001)
    half_sham = _run("sham", 0.001)

    repeat = compare_trace_records(base_contact, base_contact)
    dt = compare_trace_records(base_contact, half_contact)
    effect = compare_condition_effects(
        base_contact, base_sham, base_contact, base_sham
    )

    assert repeat.joint_or_effect_max_abs_rad == 0.0
    assert repeat.pair_active_exact is True
    assert repeat.event_interval_exact is True
    assert dt.pair_active_exact is None
    assert dt.event_midpoint_max_abs_s == pytest.approx(0.0005)
    assert effect.probe_gap_or_blocked_max_abs_m == 0.0
    assert effect.orientation_or_effect_max_abs_rad == 0.0


def test_orientation_effect_uses_project_qwxyz_pose_convention() -> None:
    # Both pairs have the same world-frame contact effect but different,
    # non-identity sham orientations.  Correct qwxyz composition cancels each
    # baseline exactly; interpreting the stored values as xyzw produces a
    # spurious nonzero cross-pair effect.
    contact_effect = _axis_angle_qwxyz((1.0, 0.0, 0.0), 0.1)
    first_sham_q = _axis_angle_qwxyz((0.0, 0.0, 1.0), 1.0)
    second_sham_q = _axis_angle_qwxyz((0.0, 0.0, 1.0), 1.2)
    first_contact_q = _reference_qwxyz_multiply(contact_effect, first_sham_q)
    second_contact_q = _reference_qwxyz_multiply(contact_effect, second_sham_q)

    delta = compare_condition_effects(
        _with_frame_orientation(_run("contact"), first_contact_q),
        _with_frame_orientation(_run("sham"), first_sham_q),
        _with_frame_orientation(_run("contact"), second_contact_q),
        _with_frame_orientation(_run("sham"), second_sham_q),
    )

    assert delta.orientation_or_effect_max_abs_rad == pytest.approx(0.0, abs=1e-15)


def test_quaternion_geodesic_is_stable_for_identical_and_tiny_rotations() -> None:
    # This exact quaternion came from the excluded local a3 repeat pair.  The
    # two serialized values were byte-identical, while the former acos(dot)
    # implementation returned 2.9802322387695312e-08 rad.
    observed = (
        0.689046833008838,
        -0.6890489505421021,
        0.15878772476618794,
        0.15878433216440335,
    )
    assert _quaternion_angle(observed, observed) == 0.0
    assert _quaternion_angle(observed, tuple(-value for value in observed)) == 0.0

    angle = 5.0e-10
    tiny = (math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0)
    assert _quaternion_angle((1.0, 0.0, 0.0, 0.0), tiny) == pytest.approx(
        angle,
        rel=0.0,
        abs=1e-20,
    )
