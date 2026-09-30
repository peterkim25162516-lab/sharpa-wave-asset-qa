"""Deterministic scientific metrics for Contact Gate C0.

Only portable observations participate in these calculations: canonical joint
states, distal-frame poses, the analytic signed sphere/plane gap, and the
frozen selected-pair activity bit.  Backend-native force payloads are never
compared across simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .records import ContactRun, ContactTraceSample


class ContactMetricError(ValueError):
    """A required C0 metric cannot be computed without guessing."""


@dataclass(frozen=True, slots=True)
class EventInterval:
    """The physics interval containing a debounced state transition."""

    first_sample_index: int
    start_s: float
    end_s: float
    midpoint_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "first_sample_index": self.first_sample_index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "midpoint_s": self.midpoint_s,
        }


@dataclass(frozen=True, slots=True)
class DebouncedActivity:
    """Debounced interval activity; sample zero is deliberately excluded."""

    required_consecutive_intervals: int
    interval_bits: tuple[bool, ...]
    onset_interval: EventInterval | None
    release_interval: EventInterval | None


@dataclass(frozen=True, slots=True)
class ContactRunMetrics:
    case_id: str
    debounce_interval_count: int
    onset_interval: EventInterval | None
    release_interval: EventInterval | None
    hold_duty: float
    recovery_duty: float
    initial_gap_m: float
    onset_gap_m: float | None
    steady_hold_mean_gap_m: float
    recovery_min_gap_m: float
    minimum_gap_m: float
    debounced_interval_bits: tuple[bool, ...]

    def to_dict(self, *, include_interval_bits: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "case_id": self.case_id,
            "debounce_interval_count": self.debounce_interval_count,
            "onset_interval": (
                None if self.onset_interval is None else self.onset_interval.to_dict()
            ),
            "release_interval": (
                None
                if self.release_interval is None
                else self.release_interval.to_dict()
            ),
            "hold_duty": self.hold_duty,
            "recovery_duty": self.recovery_duty,
            "initial_gap_m": self.initial_gap_m,
            "onset_gap_m": self.onset_gap_m,
            "steady_hold_mean_gap_m": self.steady_hold_mean_gap_m,
            "recovery_min_gap_m": self.recovery_min_gap_m,
            "minimum_gap_m": self.minimum_gap_m,
        }
        if include_interval_bits:
            payload["debounced_interval_bits"] = list(
                self.debounced_interval_bits
            )
        return payload


@dataclass(frozen=True, slots=True)
class ContactMetricDelta:
    """Frozen comparison dimensions shared by repeat, dt, and effect checks."""

    joint_or_effect_max_abs_rad: float
    probe_gap_or_blocked_max_abs_m: float
    orientation_or_effect_max_abs_rad: float
    pair_active_exact: bool | None
    event_interval_exact: bool
    event_midpoint_max_abs_s: float | None
    hold_duty_max_abs: float

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_or_effect_max_abs_rad": self.joint_or_effect_max_abs_rad,
            "probe_gap_or_blocked_max_abs_m": (
                self.probe_gap_or_blocked_max_abs_m
            ),
            "orientation_or_effect_max_abs_rad": (
                self.orientation_or_effect_max_abs_rad
            ),
            "pair_active_exact": self.pair_active_exact,
            "event_interval_exact": self.event_interval_exact,
            "event_midpoint_max_abs_s": self.event_midpoint_max_abs_s,
            "hold_duty_max_abs": self.hold_duty_max_abs,
        }


def _require_completed(run: ContactRun, context: str) -> None:
    if not isinstance(run, ContactRun):
        raise TypeError(f"{context} must be ContactRun")
    if not run.completed:
        raise ContactMetricError(f"{context} is not a completed C0 run")
    if not run.samples:
        raise ContactMetricError(f"{context} contains no samples")


def _event_interval(sample: ContactTraceSample, dt_s: float) -> EventInterval:
    if sample.step <= 0:
        raise ContactMetricError("an event cannot be assigned to sample zero")
    start = sample.time_s - dt_s
    end = sample.time_s
    return EventInterval(
        first_sample_index=sample.step,
        start_s=start,
        end_s=end,
        midpoint_s=(start + end) / 2.0,
    )


def debounce_pair_activity(
    samples: Sequence[ContactTraceSample],
    *,
    dt_s: float,
    debounce_window_s: float = 0.004,
) -> DebouncedActivity:
    """Debounce the step-ordered selected-pair bitset.

    A state change requires ``ceil(window / dt)`` consecutive physics
    intervals.  Once confirmed, the transition is backdated to the first
    candidate interval.  This preserves the protocol's event uncertainty
    interval while removing shorter pulses and dropouts.  Sample zero has no
    preceding physics interval and is not part of the returned bitset.
    """

    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("samples must be a sequence")
    if len(samples) < 2:
        raise ContactMetricError("at least sample zero and one interval are required")
    for value, name in ((dt_s, "dt_s"), (debounce_window_s, "debounce_window_s")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ContactMetricError(f"{name} must be a positive finite number")
    if not isinstance(samples[0], ContactTraceSample) or samples[0].step != 0:
        raise ContactMetricError("samples must begin with canonical sample zero")
    required = max(1, math.ceil(float(debounce_window_s) / float(dt_s) - 1e-12))
    raw: list[bool] = []
    for index, sample in enumerate(samples[1:], start=1):
        if not isinstance(sample, ContactTraceSample) or sample.step != index:
            raise ContactMetricError("samples must use a contiguous canonical step axis")
        raw.append(sample.pair_active)

    debounced = [False] * len(raw)
    state = False
    onset: EventInterval | None = None
    release: EventInterval | None = None
    cursor = 0
    while cursor < len(raw):
        candidate = raw[cursor]
        end = cursor + 1
        while end < len(raw) and raw[end] is candidate:
            end += 1
        run_length = end - cursor
        if candidate is state:
            debounced[cursor:end] = [state] * run_length
        elif run_length >= required:
            state = candidate
            debounced[cursor:end] = [state] * run_length
            interval = _event_interval(samples[cursor + 1], float(dt_s))
            if state and onset is None:
                onset = interval
            elif not state and onset is not None and release is None:
                release = interval
        else:
            debounced[cursor:end] = [state] * run_length
        cursor = end

    return DebouncedActivity(
        required_consecutive_intervals=required,
        interval_bits=tuple(debounced),
        onset_interval=onset,
        release_interval=release,
    )


def _window_sample_indices(
    samples: Sequence[ContactTraceSample],
    start_s: float,
    end_s: float,
    *,
    dt_s: float,
) -> tuple[int, ...]:
    if (
        not math.isfinite(start_s)
        or not math.isfinite(end_s)
        or start_s < 0.0
        or end_s <= start_s
    ):
        raise ContactMetricError("metric window must be finite, non-negative, and non-empty")

    def boundary_step(boundary_s: float) -> int:
        step = round(boundary_s / dt_s)
        if not math.isclose(
            boundary_s, step * dt_s, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContactMetricError(
                "metric window boundary is not aligned to the canonical step axis"
            )
        return step

    start_step = boundary_step(start_s)
    end_step = boundary_step(end_s)
    if end_step > samples[-1].step:
        raise ContactMetricError("metric window extends beyond the canonical trace")
    # Window membership is defined by canonical integer steps, not by the
    # serialized floating-point timestamp (which is only an integrity check).
    indices = tuple(range(max(1, start_step), end_step))
    if not indices:
        raise ContactMetricError(
            f"trace has no samples in frozen window [{start_s}, {end_s})"
        )
    return indices


def extract_run_metrics(
    run: ContactRun,
    *,
    debounce_window_s: float = 0.004,
    steady_hold_window_s: tuple[float, float] = (3.25, 3.75),
    recovery_window_s: tuple[float, float] = (6.75, 7.0),
) -> ContactRunMetrics:
    """Extract the C0 event, duty-cycle, and analytic-gap metrics."""

    _require_completed(run, "run")
    activity = debounce_pair_activity(
        run.samples,
        dt_s=run.case.dt_s,
        debounce_window_s=debounce_window_s,
    )
    hold_indices = _window_sample_indices(
        run.samples, *steady_hold_window_s, dt_s=run.case.dt_s
    )
    recovery_indices = _window_sample_indices(
        run.samples, *recovery_window_s, dt_s=run.case.dt_s
    )

    def activity_at_sample(index: int) -> bool:
        return activity.interval_bits[index - 1]

    hold_duty = sum(activity_at_sample(index) for index in hold_indices) / len(
        hold_indices
    )
    recovery_duty = sum(
        activity_at_sample(index) for index in recovery_indices
    ) / len(recovery_indices)
    hold_gaps = [run.samples[index].signed_gap_m for index in hold_indices]
    recovery_gaps = [run.samples[index].signed_gap_m for index in recovery_indices]
    onset_gap = (
        None
        if activity.onset_interval is None
        else run.samples[activity.onset_interval.first_sample_index].signed_gap_m
    )
    return ContactRunMetrics(
        case_id=run.case.case_id,
        debounce_interval_count=activity.required_consecutive_intervals,
        onset_interval=activity.onset_interval,
        release_interval=activity.release_interval,
        hold_duty=float(hold_duty),
        recovery_duty=float(recovery_duty),
        initial_gap_m=run.samples[0].signed_gap_m,
        onset_gap_m=onset_gap,
        steady_hold_mean_gap_m=sum(hold_gaps) / len(hold_gaps),
        recovery_min_gap_m=min(recovery_gaps),
        minimum_gap_m=min(sample.signed_gap_m for sample in run.samples),
        debounced_interval_bits=activity.interval_bits,
    )


def _aligned_samples(
    first: ContactRun, second: ContactRun
) -> tuple[tuple[ContactTraceSample, ContactTraceSample], ...]:
    _require_completed(first, "first")
    _require_completed(second, "second")
    coarse, fine = (
        (first, second)
        if first.case.dt_s >= second.case.dt_s
        else (second, first)
    )
    ratio_value = coarse.case.dt_s / fine.case.dt_s
    ratio = round(ratio_value)
    if ratio < 1 or not math.isclose(
        ratio_value, ratio, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContactMetricError("trace timesteps are not integer-aligned")
    if len(fine.samples) != (len(coarse.samples) - 1) * ratio + 1:
        raise ContactMetricError("trace durations or sample axes do not align")
    aligned: list[tuple[ContactTraceSample, ContactTraceSample]] = []
    for coarse_index, coarse_sample in enumerate(coarse.samples):
        fine_sample = fine.samples[coarse_index * ratio]
        if not math.isclose(
            coarse_sample.time_s, fine_sample.time_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ContactMetricError("trace timestamps do not align")
        aligned.append(
            (coarse_sample, fine_sample)
            if coarse is first
            else (fine_sample, coarse_sample)
        )
    return tuple(aligned)


def _quaternion_angle(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ContactMetricError("orientation must contain four quaternion values")
    first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
    second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ContactMetricError("orientation quaternion has zero norm")
    left = tuple(float(value) / first_norm for value in first)
    right = tuple(float(value) / second_norm for value in second)
    if math.fsum(a * b for a, b in zip(left, right)) < 0.0:
        right = tuple(-value for value in right)
    # For sign-aligned unit quaternions, 4*atan2(||q1-q2||, ||q1+q2||)
    # is the same SO(3) geodesic as 2*acos(|dot|), but remains accurate near
    # zero.  The acos form can report about 2.98e-8 rad for byte-identical
    # quaternions when norm roundoff leaves the normalized dot just below 1.
    difference_norm = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))
    sum_norm = math.sqrt(math.fsum((a + b) ** 2 for a, b in zip(left, right)))
    return min(math.pi, 4.0 * math.atan2(difference_norm, sum_norm))


def _quaternion_inverse(value: Sequence[float]) -> tuple[float, float, float, float]:
    # Frame poses use the project-wide ``xyz_m_qwxyz`` convention.  Keep the
    # scalar component first here as well; treating these records as xyzw
    # changes contact-minus-sham orientation effects whenever the two traces
    # have a non-identity baseline orientation.
    w, x, y, z = (float(item) for item in value)
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq <= 0.0:
        raise ContactMetricError("orientation quaternion has zero norm")
    return (w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq)


def _quaternion_multiply(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = (float(item) for item in first)
    bw, bx, by, bz = (float(item) for item in second)
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _event_midpoint_delta(
    first: ContactRunMetrics, second: ContactRunMetrics
) -> tuple[bool, float | None]:
    intervals = (
        (first.onset_interval, second.onset_interval),
        (first.release_interval, second.release_interval),
    )
    if any((left is None) != (right is None) for left, right in intervals):
        return False, None
    present = [
        (left, right)
        for left, right in intervals
        if left is not None and right is not None
    ]
    exact = all(left == right for left, right in intervals)
    maximum = max(
        (abs(left.midpoint_s - right.midpoint_s) for left, right in present),
        default=0.0,
    )
    return exact, maximum


def compare_trace_records(first: ContactRun, second: ContactRun) -> ContactMetricDelta:
    """Compare two same-condition traces on their common physical-time grid."""

    if first.case.condition != second.case.condition:
        raise ContactMetricError("trace comparison requires the same condition")
    aligned = _aligned_samples(first, second)
    joint_max = 0.0
    gap_max = 0.0
    orientation_max = 0.0
    for left, right in aligned:
        if tuple(left.joint_positions) != tuple(right.joint_positions):
            raise ContactMetricError("trace joint names or order differ")
        if tuple(left.frame_poses) != tuple(right.frame_poses):
            raise ContactMetricError("trace frame names or order differ")
        joint_max = max(
            joint_max,
            max(
                abs(left.joint_positions[name] - right.joint_positions[name])
                for name in left.joint_positions
            ),
        )
        gap_max = max(gap_max, abs(left.signed_gap_m - right.signed_gap_m))
        orientation_max = max(
            orientation_max,
            max(
                _quaternion_angle(
                    left.frame_poses[name][3:], right.frame_poses[name][3:]
                )
                for name in left.frame_poses
            ),
        )
    first_metrics = extract_run_metrics(first)
    second_metrics = extract_run_metrics(second)
    event_exact, event_delta = _event_midpoint_delta(first_metrics, second_metrics)
    pair_exact: bool | None = None
    if math.isclose(first.case.dt_s, second.case.dt_s, rel_tol=0.0, abs_tol=1e-15):
        pair_exact = tuple(sample.pair_active for sample in first.samples) == tuple(
            sample.pair_active for sample in second.samples
        )
    return ContactMetricDelta(
        joint_or_effect_max_abs_rad=joint_max,
        probe_gap_or_blocked_max_abs_m=gap_max,
        orientation_or_effect_max_abs_rad=orientation_max,
        pair_active_exact=pair_exact,
        event_interval_exact=event_exact,
        event_midpoint_max_abs_s=event_delta,
        hold_duty_max_abs=abs(first_metrics.hold_duty - second_metrics.hold_duty),
    )


def compare_condition_effects(
    first_contact: ContactRun,
    first_sham: ContactRun,
    second_contact: ContactRun,
    second_sham: ContactRun,
) -> ContactMetricDelta:
    """Compare contact-minus-sham effects between repeats, dt levels, or engines."""

    for contact, sham, label in (
        (first_contact, first_sham, "first"),
        (second_contact, second_sham, "second"),
    ):
        if contact.case.condition != "contact" or sham.case.condition != "sham":
            raise ContactMetricError(f"{label} effect is not contact-minus-sham")
        if (
            contact.case.simulator != sham.case.simulator
            or contact.case.hand != sham.case.hand
            or contact.case.timestep_variant != sham.case.timestep_variant
            or contact.case.repeat_index != sham.case.repeat_index
        ):
            raise ContactMetricError(f"{label} contact and sham cases are not matched")

    first_pairs = _aligned_samples(first_contact, first_sham)
    second_pairs = _aligned_samples(second_contact, second_sham)
    # The two effects may have base versus halved dt.  Align their contact
    # traces, then index each matched sham at the same physical timestamp.
    cross_contact = _aligned_samples(first_contact, second_contact)
    first_by_time = {
        round(contact.time_s, 9): (contact, sham)
        for contact, sham in first_pairs
    }
    second_by_time = {
        round(contact.time_s, 9): (contact, sham)
        for contact, sham in second_pairs
    }
    joint_max = 0.0
    blocked_max = 0.0
    orientation_max = 0.0
    for left_contact, right_contact in cross_contact:
        key = round(left_contact.time_s, 9)
        if key not in first_by_time or key not in second_by_time:
            raise ContactMetricError("condition effects do not share a physical-time grid")
        left_contact, left_sham = first_by_time[key]
        right_contact, right_sham = second_by_time[key]
        names = tuple(left_contact.joint_positions)
        if not (
            names
            == tuple(left_sham.joint_positions)
            == tuple(right_contact.joint_positions)
            == tuple(right_sham.joint_positions)
        ):
            raise ContactMetricError("condition-effect joint mappings differ")
        joint_max = max(
            joint_max,
            max(
                abs(
                    (left_contact.joint_positions[name] - left_sham.joint_positions[name])
                    - (
                        right_contact.joint_positions[name]
                        - right_sham.joint_positions[name]
                    )
                )
                for name in names
            ),
        )
        left_blocked = left_contact.signed_gap_m - left_sham.signed_gap_m
        right_blocked = right_contact.signed_gap_m - right_sham.signed_gap_m
        blocked_max = max(blocked_max, abs(left_blocked - right_blocked))
        frame_names = tuple(left_contact.frame_poses)
        if not (
            frame_names
            == tuple(left_sham.frame_poses)
            == tuple(right_contact.frame_poses)
            == tuple(right_sham.frame_poses)
        ):
            raise ContactMetricError("condition-effect frame mappings differ")
        for name in frame_names:
            left_effect = _quaternion_multiply(
                left_contact.frame_poses[name][3:],
                _quaternion_inverse(left_sham.frame_poses[name][3:]),
            )
            right_effect = _quaternion_multiply(
                right_contact.frame_poses[name][3:],
                _quaternion_inverse(right_sham.frame_poses[name][3:]),
            )
            orientation_max = max(
                orientation_max, _quaternion_angle(left_effect, right_effect)
            )

    first_metrics = extract_run_metrics(first_contact)
    second_metrics = extract_run_metrics(second_contact)
    event_exact, event_delta = _event_midpoint_delta(first_metrics, second_metrics)
    pair_exact: bool | None = None
    if math.isclose(
        first_contact.case.dt_s,
        second_contact.case.dt_s,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        first_effect_bits = tuple(
            contact.pair_active != sham.pair_active
            for contact, sham in zip(first_contact.samples, first_sham.samples)
        )
        second_effect_bits = tuple(
            contact.pair_active != sham.pair_active
            for contact, sham in zip(second_contact.samples, second_sham.samples)
        )
        pair_exact = first_effect_bits == second_effect_bits
    return ContactMetricDelta(
        joint_or_effect_max_abs_rad=joint_max,
        probe_gap_or_blocked_max_abs_m=blocked_max,
        orientation_or_effect_max_abs_rad=orientation_max,
        pair_active_exact=pair_exact,
        event_interval_exact=event_exact,
        event_midpoint_max_abs_s=event_delta,
        hold_duty_max_abs=abs(first_metrics.hold_duty - second_metrics.hold_duty),
    )


def maximum_delta(deltas: Iterable[ContactMetricDelta]) -> ContactMetricDelta:
    """Reduce a non-empty set of deltas without hiding unavailable events."""

    values = tuple(deltas)
    if not values:
        raise ContactMetricError("at least one delta is required")
    midpoint_values = [
        value.event_midpoint_max_abs_s
        for value in values
        if value.event_midpoint_max_abs_s is not None
    ]
    midpoint = (
        None
        if len(midpoint_values) != len(values)
        else max(midpoint_values, default=0.0)
    )
    pair_values = [value.pair_active_exact for value in values]
    pair_exact = (
        None if any(value is None for value in pair_values) else all(pair_values)
    )
    return ContactMetricDelta(
        joint_or_effect_max_abs_rad=max(
            value.joint_or_effect_max_abs_rad for value in values
        ),
        probe_gap_or_blocked_max_abs_m=max(
            value.probe_gap_or_blocked_max_abs_m for value in values
        ),
        orientation_or_effect_max_abs_rad=max(
            value.orientation_or_effect_max_abs_rad for value in values
        ),
        pair_active_exact=pair_exact,
        event_interval_exact=all(value.event_interval_exact for value in values),
        event_midpoint_max_abs_s=midpoint,
        hold_duty_max_abs=max(value.hold_duty_max_abs for value in values),
    )


# Explicit alias used by prose and lightweight consumers.
extract_contact_metrics = extract_run_metrics


__all__ = [
    "ContactMetricDelta",
    "ContactMetricError",
    "ContactRunMetrics",
    "DebouncedActivity",
    "EventInterval",
    "compare_condition_effects",
    "compare_trace_records",
    "debounce_pair_activity",
    "extract_contact_metrics",
    "extract_run_metrics",
    "maximum_delta",
]
