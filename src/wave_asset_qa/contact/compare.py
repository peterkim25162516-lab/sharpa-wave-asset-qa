"""Fail-closed evidence validation and scientific classification for C0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping

from .contracts import ContactManifest
from .metrics import (
    ContactMetricDelta,
    ContactMetricError,
    ContactRunMetrics,
    compare_condition_effects,
    compare_trace_records,
    extract_run_metrics,
    maximum_delta,
)
from .records import ContactCaseRecord, ContactRun
from .scenarios import contact_manifest_sha256, expand_contact_cases


class EvidenceStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


class ScienceStatus(str, Enum):
    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
    DIVERGENT = "DIVERGENT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class ContactCheck:
    stage: str
    check_id: str
    passed: bool
    observed: object
    criterion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "check_id": self.check_id,
            "passed": self.passed,
            "observed": self.observed,
            "criterion": self.criterion,
        }


@dataclass(frozen=True, slots=True)
class ContactGateResult:
    manifest_id: str
    manifest_sha256: str
    expected_run_count: int
    received_run_count: int
    completed_run_count: int
    evidence_status: EvidenceStatus
    science_status: ScienceStatus
    metrics: tuple[ContactRunMetrics, ...]
    checks: tuple[ContactCheck, ...]
    invalid_reasons: tuple[str, ...]
    scientific_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "expected_run_count": self.expected_run_count,
            "received_run_count": self.received_run_count,
            "completed_run_count": self.completed_run_count,
            "evidence_status": self.evidence_status.value,
            "science_status": self.science_status.value,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "checks": [check.to_dict() for check in self.checks],
            "invalid_reasons": list(self.invalid_reasons),
            "scientific_reasons": list(self.scientific_reasons),
        }


def _expected_case_record(case: object) -> ContactCaseRecord:
    to_dict = getattr(case, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("expanded case does not expose to_dict")
    return ContactCaseRecord.from_dict(to_dict())


def _case_key(run: ContactRun, *fields: str) -> tuple[object, ...]:
    return tuple(getattr(run.case, field) for field in fields)


def _check(
    stage: str,
    check_id: str,
    observed: object,
    passed: bool,
    criterion: str,
) -> ContactCheck:
    return ContactCheck(
        stage=stage,
        check_id=check_id,
        passed=bool(passed),
        observed=observed,
        criterion=criterion,
    )


def _aggregate_deltas(deltas: Iterable[ContactMetricDelta]) -> ContactMetricDelta:
    return maximum_delta(tuple(deltas))


def _backend_contact_presence(
    runs: Mapping[str, ContactRun],
) -> dict[str, bool]:
    """Return raw selected-pair presence for each frozen backend."""

    return {
        backend: any(
            sample.pair_active
            for run in runs.values()
            if run.case.simulator == backend and run.case.condition == "contact"
            for sample in run.samples
        )
        for backend in ("mujoco", "ovphysx")
    }


def _repeat_checks(
    manifest: ContactManifest,
    runs: Mapping[str, ContactRun],
) -> tuple[ContactCheck, ...]:
    groups: dict[tuple[object, ...], list[ContactRun]] = {}
    for run in runs.values():
        groups.setdefault(
            _case_key(
                run,
                "simulator",
                "hand",
                "condition",
                "timestep_variant",
            ),
            [],
        ).append(run)
    trace_deltas: list[ContactMetricDelta] = []
    for values in groups.values():
        ordered = sorted(values, key=lambda item: item.case.repeat_index)
        if len(ordered) != 2:
            raise ContactMetricError("repeat group does not contain exactly r01/r02")
        trace_deltas.append(compare_trace_records(ordered[0], ordered[1]))

    effects: dict[tuple[object, ...], dict[tuple[str, int], ContactRun]] = {}
    for run in runs.values():
        effects.setdefault(
            _case_key(run, "simulator", "hand", "timestep_variant"), {}
        )[(run.case.condition, run.case.repeat_index)] = run
    effect_deltas = []
    for values in effects.values():
        effect_deltas.append(
            compare_condition_effects(
                values[("contact", 1)],
                values[("sham", 1)],
                values[("contact", 2)],
                values[("sham", 2)],
            )
        )
    observed = _aggregate_deltas((*trace_deltas, *effect_deltas))
    limits = manifest.thresholds.repeatability
    return (
        _check(
            "repeatability",
            "joint_or_effect_max_abs_rad",
            observed.joint_or_effect_max_abs_rad,
            observed.joint_or_effect_max_abs_rad
            <= limits.joint_or_effect_max_abs_rad,
            f"<= {limits.joint_or_effect_max_abs_rad}",
        ),
        _check(
            "repeatability",
            "probe_gap_or_blocked_max_abs_m",
            observed.probe_gap_or_blocked_max_abs_m,
            observed.probe_gap_or_blocked_max_abs_m
            <= limits.probe_gap_or_blocked_max_abs_m,
            f"<= {limits.probe_gap_or_blocked_max_abs_m}",
        ),
        _check(
            "repeatability",
            "orientation_or_effect_max_abs_rad",
            observed.orientation_or_effect_max_abs_rad,
            observed.orientation_or_effect_max_abs_rad
            <= limits.orientation_or_effect_max_abs_rad,
            f"<= {limits.orientation_or_effect_max_abs_rad}",
        ),
        _check(
            "repeatability",
            "pair_active_exact",
            observed.pair_active_exact,
            observed.pair_active_exact is limits.pair_active_exact,
            "must be exactly true for complete N+1 bitsets",
        ),
        _check(
            "repeatability",
            "event_interval_exact",
            observed.event_interval_exact,
            observed.event_interval_exact is limits.event_interval_exact,
            "must be exactly true",
        ),
    )


def _dt_checks(
    manifest: ContactManifest,
    runs: Mapping[str, ContactRun],
) -> tuple[ContactCheck, ...]:
    groups: dict[tuple[object, ...], dict[str, ContactRun]] = {}
    for run in runs.values():
        groups.setdefault(
            _case_key(run, "simulator", "hand", "condition", "repeat_index"),
            {},
        )[run.case.timestep_variant] = run
    trace_deltas = [
        compare_trace_records(values["base"], values["halved"])
        for values in groups.values()
    ]
    effects: dict[
        tuple[object, ...], dict[tuple[str, str], ContactRun]
    ] = {}
    for run in runs.values():
        effects.setdefault(
            _case_key(run, "simulator", "hand", "repeat_index"), {}
        )[(run.case.condition, run.case.timestep_variant)] = run
    effect_deltas = [
        compare_condition_effects(
            values[("contact", "base")],
            values[("sham", "base")],
            values[("contact", "halved")],
            values[("sham", "halved")],
        )
        for values in effects.values()
    ]
    observed = _aggregate_deltas((*trace_deltas, *effect_deltas))
    limits = manifest.thresholds.dt_halving
    checks = [
        _check("dt_halving", name, actual, actual <= limit, f"<= {limit}")
        for name, actual, limit in (
            (
                "joint_or_effect_max_abs_rad",
                observed.joint_or_effect_max_abs_rad,
                limits.joint_or_effect_max_abs_rad,
            ),
            (
                "probe_gap_or_blocked_max_abs_m",
                observed.probe_gap_or_blocked_max_abs_m,
                limits.probe_gap_or_blocked_max_abs_m,
            ),
            (
                "orientation_or_effect_max_abs_rad",
                observed.orientation_or_effect_max_abs_rad,
                limits.orientation_or_effect_max_abs_rad,
            ),
            (
                "hold_duty_max_abs",
                observed.hold_duty_max_abs,
                limits.hold_duty_max_abs,
            ),
        )
    ]
    checks.append(
        _check(
            "dt_halving",
            "event_midpoint_max_abs_s",
            observed.event_midpoint_max_abs_s,
            observed.event_midpoint_max_abs_s is not None
            and observed.event_midpoint_max_abs_s
            <= limits.event_midpoint_max_abs_s,
            f"<= {limits.event_midpoint_max_abs_s} and must be available",
        )
    )
    return tuple(checks)


def _crosssim_checks(
    manifest: ContactManifest,
    runs: Mapping[str, ContactRun],
    metrics: Mapping[str, ContactRunMetrics],
) -> tuple[ContactCheck, ...]:
    groups: dict[tuple[object, ...], dict[str, ContactRun]] = {}
    for run in runs.values():
        groups.setdefault(
            _case_key(
                run,
                "hand",
                "condition",
                "timestep_variant",
                "repeat_index",
            ),
            {},
        )[run.case.simulator] = run
    event_deltas: list[float] = []
    hold_deltas: list[float] = []
    onset_gap_deltas: list[float] = []
    steady_gap_deltas: list[float] = []
    for values in groups.values():
        left = metrics[values["mujoco"].case.case_id]
        right = metrics[values["ovphysx"].case.case_id]
        hold_deltas.append(abs(left.hold_duty - right.hold_duty))
        if values["mujoco"].case.condition == "contact":
            # The sham trajectory is a baseline for the difference-in-
            # differences effects below.  Comparing its raw gap directly
            # would re-label a contact-free baseline offset as contact
            # divergence, which C0 explicitly forbids.
            steady_gap_deltas.append(
                abs(left.steady_hold_mean_gap_m - right.steady_hold_mean_gap_m)
            )
            for first_event, second_event in (
                (left.onset_interval, right.onset_interval),
                (left.release_interval, right.release_interval),
            ):
                if first_event is None or second_event is None:
                    raise ContactMetricError(
                        "contact event is unavailable for cross-simulator comparison"
                    )
                event_deltas.append(
                    abs(first_event.midpoint_s - second_event.midpoint_s)
                )
            if left.onset_gap_m is None or right.onset_gap_m is None:
                raise ContactMetricError(
                    "contact onset gap is unavailable for cross-simulator comparison"
                )
            onset_gap_deltas.append(abs(left.onset_gap_m - right.onset_gap_m))

    effects: dict[
        tuple[object, ...], dict[tuple[str, str], ContactRun]
    ] = {}
    for run in runs.values():
        effects.setdefault(
            _case_key(run, "hand", "timestep_variant", "repeat_index"), {}
        )[(run.case.simulator, run.case.condition)] = run
    effect_delta = _aggregate_deltas(
        compare_condition_effects(
            values[("mujoco", "contact")],
            values[("mujoco", "sham")],
            values[("ovphysx", "contact")],
            values[("ovphysx", "sham")],
        )
        for values in effects.values()
    )
    observed = {
        "event_midpoint_max_abs_s": max(event_deltas),
        "hold_duty_max_abs": max(hold_deltas),
        "onset_gap_max_abs_m": max(onset_gap_deltas),
        "steady_hold_mean_gap_max_abs_m": max(steady_gap_deltas),
        "joint_effect_max_abs_rad": effect_delta.joint_or_effect_max_abs_rad,
        "blocked_travel_max_abs_m": effect_delta.probe_gap_or_blocked_max_abs_m,
        "orientation_effect_max_abs_rad": (
            effect_delta.orientation_or_effect_max_abs_rad
        ),
    }
    limits = manifest.thresholds.cross_simulator.to_dict()
    return tuple(
        _check(
            "cross_simulator",
            name,
            actual,
            actual <= float(limits[name]),
            f"<= {limits[name]}",
        )
        for name, actual in observed.items()
    )


def evaluate_contact_c0(
    manifest: ContactManifest, collected_runs: Iterable[ContactRun]
) -> ContactGateResult:
    """Validate the closed 32-case evidence set, then classify its science.

    Structural, capability, provenance, completion, or metric availability
    failures produce ``EvidenceStatus.INVALID`` and
    ``ScienceStatus.INCONCLUSIVE``.  ``DIVERGENT`` is reserved for a valid
    evidence set whose admission/repeat/dt prerequisites pass and whose
    cross-simulator diagnostic thresholds are exceeded.
    """

    if not isinstance(manifest, ContactManifest):
        raise TypeError("manifest must be ContactManifest")
    expected_cases = expand_contact_cases(manifest)
    expected = {case.case_id: case for case in expected_cases}
    manifest_digest = contact_manifest_sha256(manifest)
    by_id: dict[str, ContactRun] = {}
    invalid: list[str] = []
    received = 0
    for index, run in enumerate(collected_runs):
        received += 1
        if not isinstance(run, ContactRun):
            invalid.append(f"received[{index}] is not a ContactRun")
            continue
        case_id = run.case.case_id
        if case_id in by_id:
            invalid.append(f"duplicate case: {case_id}")
            continue
        by_id[case_id] = run
        expected_case = expected.get(case_id)
        if expected_case is None:
            invalid.append(f"unexpected case: {case_id}")
            continue
        if run.case != _expected_case_record(expected_case):
            invalid.append(f"case identity differs from manifest: {case_id}")
        if run.manifest_sha256 != manifest_digest:
            invalid.append(f"manifest digest mismatch: {case_id}")
        if not run.completed:
            invalid.append(f"case did not complete: {case_id}")
        provenance = run.provenance
        if provenance.get("asset_commit") != manifest.provenance.commit:
            invalid.append(f"asset commit mismatch: {case_id}")
        if provenance.get("asset_git_tree") != manifest.provenance.asset_git_tree:
            invalid.append(f"asset tree mismatch: {case_id}")

    missing = sorted(set(expected) - set(by_id))
    if missing:
        invalid.append("missing cases: " + ", ".join(missing))
    completed = sum(run.completed for run in by_id.values())
    if len(by_id) == len(expected):
        revisions = {run.provenance.get("source_revision") for run in by_id.values()}
        trees = {run.provenance.get("source_tree") for run in by_id.values()}
        process_ids = {
            run.provenance.get("fresh_process_identity_sha256")
            for run in by_id.values()
        }
        if len(revisions) != 1 or len(trees) != 1:
            invalid.append("source revision/tree identity changed across the matrix")
        if len(process_ids) != len(expected):
            invalid.append("fresh process identities are not unique for all 32 cases")

    if invalid:
        return ContactGateResult(
            manifest_id=manifest.manifest_id,
            manifest_sha256=manifest_digest,
            expected_run_count=len(expected),
            received_run_count=received,
            completed_run_count=completed,
            evidence_status=EvidenceStatus.INVALID,
            science_status=ScienceStatus.INCONCLUSIVE,
            metrics=(),
            checks=(),
            invalid_reasons=tuple(invalid),
            scientific_reasons=("invalid_evidence",),
        )

    try:
        metric_by_id = {
            case_id: extract_run_metrics(run) for case_id, run in by_id.items()
        }
        admission = manifest.thresholds.admission
        contact_metrics = [
            metric
            for case_id, metric in metric_by_id.items()
            if by_id[case_id].case.condition == "contact"
        ]
        sham_metrics = [
            metric
            for case_id, metric in metric_by_id.items()
            if by_id[case_id].case.condition == "sham"
        ]
        checks: list[ContactCheck] = [
            _check(
                "admission",
                "initial_gap_min_m",
                min(item.initial_gap_m for item in metric_by_id.values()),
                all(
                    item.initial_gap_m >= admission.initial_gap_min_m
                    for item in metric_by_id.values()
                ),
                f">= {admission.initial_gap_min_m} in every run",
            ),
            _check(
                "admission",
                "recovery_gap_min_m",
                min(item.recovery_min_gap_m for item in metric_by_id.values()),
                all(
                    item.recovery_min_gap_m >= admission.recovery_gap_min_m
                    for item in metric_by_id.values()
                ),
                f">= {admission.recovery_gap_min_m} in every run",
            ),
            _check(
                "admission",
                "sham_hold_mean_gap_max_m",
                max(item.steady_hold_mean_gap_m for item in sham_metrics),
                all(
                    item.steady_hold_mean_gap_m
                    <= admission.sham_hold_mean_gap_max_m
                    for item in sham_metrics
                ),
                f"<= {admission.sham_hold_mean_gap_max_m} in every sham run",
            ),
            _check(
                "admission",
                "condition_signal_semantics",
                {
                    "sham_never_active": all(
                        item.onset_interval is None
                        and item.hold_duty == 0.0
                        for item in sham_metrics
                    ),
                    "recovery_inactive": all(
                        item.recovery_duty == 0.0 for item in metric_by_id.values()
                    ),
                },
                all(item.recovery_duty == 0.0 for item in contact_metrics)
                and all(
                    item.onset_interval is None
                    and item.hold_duty == 0.0
                    and item.recovery_duty == 0.0
                    for item in sham_metrics
                ),
                "sham and recovery stay inactive",
            ),
        ]
        repeat_checks = _repeat_checks(manifest, by_id)
        dt_checks = _dt_checks(manifest, by_id)
        checks.extend(repeat_checks)
        checks.extend(dt_checks)
        prerequisite_passed = all(check.passed for check in checks)
        cross_checks: tuple[ContactCheck, ...] = ()
        scientific_reasons: tuple[str, ...] = ()
        if prerequisite_passed:
            presence = _backend_contact_presence(by_id)
            present_count = sum(presence.values())
            checks.append(
                _check(
                    "admission",
                    "selected_pair_exercised",
                    presence,
                    present_count > 0,
                    "at least one backend must observe the selected pair",
                )
            )
            if present_count == 0:
                science = ScienceStatus.INCONCLUSIVE
                scientific_reasons = ("contact_not_exercised",)
            elif present_count == 1:
                checks.append(
                    _check(
                        "cross_simulator",
                        "contact_presence_match",
                        presence,
                        False,
                        "both backends must agree on selected-pair presence",
                    )
                )
                science = ScienceStatus.DIVERGENT
                scientific_reasons = ("contact_presence_mismatch",)
            else:
                continuous_checks = (
                    _check(
                        "admission",
                        "contact_hold_duty_min",
                        min(item.hold_duty for item in contact_metrics),
                        all(
                            item.hold_duty >= admission.contact_hold_duty_min
                            for item in contact_metrics
                        ),
                        (
                            f">= {admission.contact_hold_duty_min} in every "
                            "contact run"
                        ),
                    ),
                    _check(
                        "admission",
                        "contact_events_complete",
                        all(
                            item.onset_interval is not None
                            and item.release_interval is not None
                            for item in contact_metrics
                        ),
                        all(
                            item.onset_interval is not None
                            and item.release_interval is not None
                            for item in contact_metrics
                        ),
                        "every contact run has debounced onset and release",
                    ),
                )
                checks.extend(continuous_checks)
                if not all(check.passed for check in continuous_checks):
                    science = ScienceStatus.INCONCLUSIVE
                    scientific_reasons = ("continuous_contact_not_admitted",)
                else:
                    cross_checks = _crosssim_checks(
                        manifest, by_id, metric_by_id
                    )
                    checks.extend(cross_checks)
                    failed_cross = tuple(
                        check.check_id for check in cross_checks if not check.passed
                    )
                    if failed_cross:
                        science = ScienceStatus.DIVERGENT
                        scientific_reasons = tuple(
                            f"cross_simulator_threshold_exceeded:{check_id}"
                            for check_id in failed_cross
                        )
                    else:
                        science = ScienceStatus.WITHIN_TOLERANCE
        else:
            science = ScienceStatus.INCONCLUSIVE
            scientific_reasons = ("admission_repeatability_or_dt_failed",)
    except (ContactMetricError, KeyError, ValueError, ZeroDivisionError) as exc:
        return ContactGateResult(
            manifest_id=manifest.manifest_id,
            manifest_sha256=manifest_digest,
            expected_run_count=len(expected),
            received_run_count=received,
            completed_run_count=completed,
            evidence_status=EvidenceStatus.INVALID,
            science_status=ScienceStatus.INCONCLUSIVE,
            metrics=(),
            checks=(),
            invalid_reasons=(f"scientific metric extraction failed: {exc}",),
            scientific_reasons=("invalid_evidence",),
        )

    return ContactGateResult(
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest_digest,
        expected_run_count=len(expected),
        received_run_count=received,
        completed_run_count=completed,
        evidence_status=EvidenceStatus.VALID,
        science_status=science,
        metrics=tuple(metric_by_id[case.case_id] for case in expected_cases),
        checks=tuple(checks),
        invalid_reasons=(),
        scientific_reasons=scientific_reasons,
    )


# Compatibility with the Gate 0 naming style.
compare_contact_c0_runs = evaluate_contact_c0


__all__ = [
    "ContactCheck",
    "ContactGateResult",
    "EvidenceStatus",
    "ScienceStatus",
    "compare_contact_c0_runs",
    "evaluate_contact_c0",
]
